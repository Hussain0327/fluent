"""Audio bridge — Twilio Media Streams <-> PersonaPlex WebSocket.

Twilio sends JSON messages over WebSocket with base64-encoded mulaw audio.
PersonaPlex speaks binary Opus frames with a single-byte type prefix.

This bridge transcodes between the two formats in real time.
"""

from __future__ import annotations

import asyncio
import base64
import json
import uuid
from urllib.parse import urlencode

import aiohttp
import numpy as np

from gateway.config import settings
from gateway.db.connection import get_pool
from gateway.memory.extraction import process_conversation
from gateway.memory.retrieval import build_memory_context
from gateway.memory.store import (
    add_message,
    create_conversation,
    end_conversation,
    get_or_create_user,
)
from gateway.text import llm_client
from gateway.utils.logging import get_logger
from gateway.voice.transcript import TranscriptCapture
from gateway.voice.transcoder import mulaw_8k_to_pcm_24k, pcm_24k_to_mulaw_8k

log = get_logger(__name__)

# PersonaPlex sample rate — Mimi model operates at 24kHz
PP_SAMPLE_RATE = 24000

# Opus frame duration for PersonaPlex (matches sphn defaults)
OPUS_FRAME_MS = 20
OPUS_FRAME_SAMPLES = PP_SAMPLE_RATE * OPUS_FRAME_MS // 1000  # 480 samples


class VoiceBridge:
    """Bridges a Twilio Media Streams WebSocket to a PersonaPlex WebSocket."""

    def __init__(self) -> None:
        self._twilio_ws = None
        self._pp_ws = None
        self._session: aiohttp.ClientSession | None = None
        self._transcript = TranscriptCapture()
        self._close = False
        self._stream_sid: str | None = None
        self._conversation_id: uuid.UUID | None = None
        self._user_id: uuid.UUID | None = None
        # Buffer for accumulating PCM samples before Opus encoding
        self._pcm_buffer = np.array([], dtype=np.float32)

    async def start(
        self,
        twilio_ws,
        caller_phone: str,
        voice_prompt: str | None = None,
        text_prompt_override: str | None = None,
    ) -> None:
        """Start the bridge between Twilio and PersonaPlex."""
        self._twilio_ws = twilio_ws
        voice_prompt = voice_prompt or settings.default_voice_prompt

        pool = await get_pool()
        async with pool.acquire() as conn:
            user = await get_or_create_user(conn, caller_phone)
            self._user_id = user["id"]

            # Build memory context for voice prompt
            memory_context = await build_memory_context(
                conn, self._user_id, "voice conversation"
            )

            # Create conversation record
            conv = await create_conversation(conn, self._user_id, "voice")
            self._conversation_id = conv["id"]

        # Build text prompt with injected memories
        base_prompt = text_prompt_override or (
            "You are a helpful, friendly AI assistant having a voice conversation. "
            "Be natural and conversational."
        )
        if memory_context:
            text_prompt = f"{base_prompt}\n\n{memory_context}"
        else:
            text_prompt = base_prompt

        # Build PersonaPlex WebSocket URL
        params = {
            "voice_prompt": voice_prompt,
            "text_prompt": text_prompt,
        }
        pp_url = f"{settings.personaplex_ws_url}?{urlencode(params)}"

        log.info(
            "bridge_starting",
            user_id=str(self._user_id),
            conversation_id=str(self._conversation_id),
            voice_prompt=voice_prompt,
        )

        # Connect to PersonaPlex
        self._session = aiohttp.ClientSession()
        try:
            self._pp_ws = await self._session.ws_connect(pp_url)
        except Exception:
            log.exception("personaplex_connect_failed")
            await self._session.close()
            raise

        # Wait for handshake (0x00 byte)
        handshake_msg = await self._pp_ws.receive()
        if handshake_msg.type == aiohttp.WSMsgType.BINARY:
            if handshake_msg.data[0] == 0x00:
                log.info("personaplex_handshake_received")
            else:
                log.warning("unexpected_first_message", kind=handshake_msg.data[0])
        else:
            log.error("handshake_failed", msg_type=str(handshake_msg.type))
            await self._cleanup()
            return

        # Run bridge loops concurrently
        try:
            tasks = [
                asyncio.create_task(self._twilio_to_personaplex()),
                asyncio.create_task(self._personaplex_to_twilio()),
            ]
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            self._close = True
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        finally:
            await self._on_disconnect()
            await self._cleanup()

    async def _twilio_to_personaplex(self) -> None:
        """Receive Twilio media stream messages, transcode, forward to PersonaPlex."""
        async for msg in self._twilio_ws:
            if self._close:
                return
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                event = data.get("event")
                if event == "media":
                    payload_b64 = data["media"]["payload"]
                    mulaw_bytes = base64.b64decode(payload_b64)
                    # Transcode mulaw 8kHz -> PCM 24kHz
                    pcm_24k = mulaw_8k_to_pcm_24k(mulaw_bytes)
                    # Send as binary 0x01 + raw PCM data
                    # PersonaPlex expects Opus — we need to send through its
                    # OpusStreamReader, which means sending raw Opus frames.
                    # However, the server's opus_reader.append_bytes() expects
                    # raw Opus packet bytes. We must Opus-encode our PCM data.
                    await self._send_pcm_to_personaplex(pcm_24k)
                elif event == "start":
                    self._stream_sid = data.get("start", {}).get("streamSid")
                    log.info("twilio_stream_started", stream_sid=self._stream_sid)
                elif event == "stop":
                    log.info("twilio_stream_stopped")
                    return
            elif msg.type in (
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.ERROR,
            ):
                return

    async def _send_pcm_to_personaplex(self, pcm_24k: np.ndarray) -> None:
        """Buffer PCM and send Opus-encoded frames to PersonaPlex.

        PersonaPlex's server uses sphn.OpusStreamReader.append_bytes() which
        expects raw Opus packet data. We use the same library to encode.
        """
        if self._pp_ws is None or self._pp_ws.closed:
            return
        # For now, send raw PCM bytes prefixed with 0x01.
        # The PersonaPlex server will read this via opus_reader.append_bytes().
        # We need to match what the client sends: Opus-encoded audio bytes.
        #
        # Since we can't use sphn directly (it requires torch + CUDA),
        # we encode with opuslib or send raw bytes and let the Opus reader handle it.
        # The sphn OpusStreamReader handles raw Opus frames.
        #
        # We'll accumulate PCM and encode to Opus frames using the ctypes-based
        # Opus encoder built into this service. For v1, we send the PCM data
        # in the format that PersonaPlex's opus_reader expects.
        self._pcm_buffer = np.concatenate([self._pcm_buffer, pcm_24k])

        # Opus encode in 20ms frames (480 samples at 24kHz)
        while len(self._pcm_buffer) >= OPUS_FRAME_SAMPLES:
            frame = self._pcm_buffer[:OPUS_FRAME_SAMPLES]
            self._pcm_buffer = self._pcm_buffer[OPUS_FRAME_SAMPLES:]
            opus_bytes = self._opus_encode(frame)
            if opus_bytes:
                await self._pp_ws.send_bytes(b"\x01" + opus_bytes)

    def _opus_encode(self, pcm_frame: np.ndarray) -> bytes | None:
        """Encode a single PCM frame to Opus.

        Uses ctypes binding to libopus for encoding 24kHz mono audio.
        """
        if not hasattr(self, "_encoder"):
            self._init_opus_encoder()
        import ctypes

        # Convert float32 [-1,1] to int16
        int16_data = (np.clip(pcm_frame, -1.0, 1.0) * 32767).astype(np.int16)
        in_buf = int16_data.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        out_buf = (ctypes.c_char * 4000)()
        n = self._opus_lib.opus_encode(
            self._encoder, in_buf, OPUS_FRAME_SAMPLES, out_buf, 4000
        )
        if n < 0:
            return None
        return bytes(out_buf[:n])

    def _opus_decode(self, opus_data: bytes) -> np.ndarray | None:
        """Decode Opus packet to PCM float32."""
        if not hasattr(self, "_decoder"):
            self._init_opus_decoder()
        import ctypes

        out_buf = (ctypes.c_int16 * OPUS_FRAME_SAMPLES)()
        n = self._opus_lib.opus_decode(
            self._decoder,
            opus_data,
            len(opus_data),
            out_buf,
            OPUS_FRAME_SAMPLES,
            0,
        )
        if n < 0:
            return None
        pcm_int16 = np.frombuffer(out_buf, dtype=np.int16, count=n)
        return pcm_int16.astype(np.float32) / 32768.0

    def _init_opus_encoder(self) -> None:
        import ctypes
        import ctypes.util

        lib_path = ctypes.util.find_library("opus")
        self._opus_lib = ctypes.cdll.LoadLibrary(lib_path)
        err = ctypes.c_int(0)
        self._encoder = self._opus_lib.opus_encoder_create(
            PP_SAMPLE_RATE, 1, 2048, ctypes.byref(err)  # OPUS_APPLICATION_AUDIO
        )

    def _init_opus_decoder(self) -> None:
        import ctypes
        import ctypes.util

        if not hasattr(self, "_opus_lib"):
            lib_path = ctypes.util.find_library("opus")
            self._opus_lib = ctypes.cdll.LoadLibrary(lib_path)
        err = ctypes.c_int(0)
        self._decoder = self._opus_lib.opus_decoder_create(
            PP_SAMPLE_RATE, 1, ctypes.byref(err)
        )

    async def _personaplex_to_twilio(self) -> None:
        """Receive PersonaPlex messages, transcode audio, forward to Twilio."""
        async for msg in self._pp_ws:
            if self._close:
                return
            if msg.type == aiohttp.WSMsgType.BINARY:
                data = msg.data
                if len(data) == 0:
                    continue
                kind = data[0]
                payload = data[1:]

                if kind == 0x01:  # Audio (Opus)
                    pcm_24k = self._opus_decode(payload)
                    if pcm_24k is not None:
                        mulaw_8k = pcm_24k_to_mulaw_8k(pcm_24k)
                        await self._send_mulaw_to_twilio(mulaw_8k)

                elif kind == 0x02:  # Text token
                    text = payload.decode("utf-8", errors="replace")
                    self._transcript.add_token(text)

                elif kind == 0x00:  # Duplicate handshake (ignore)
                    pass

            elif msg.type in (
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.ERROR,
            ):
                return

    async def _send_mulaw_to_twilio(self, mulaw_bytes: bytes) -> None:
        """Send mulaw audio back to Twilio as a media message."""
        if self._twilio_ws is None or self._twilio_ws.closed:
            return
        payload_b64 = base64.b64encode(mulaw_bytes).decode("ascii")
        message = {
            "event": "media",
            "streamSid": self._stream_sid,
            "media": {"payload": payload_b64},
        }
        await self._twilio_ws.send_str(json.dumps(message))

    async def _on_disconnect(self) -> None:
        """Handle disconnection — save transcript and trigger fact extraction."""
        log.info(
            "bridge_disconnected",
            conversation_id=str(self._conversation_id),
        )

        if self._conversation_id is None or self._user_id is None:
            return

        pool = await get_pool()

        # Save transcript as messages
        transcript = self._transcript.get_transcript()
        if transcript:
            async with pool.acquire() as conn:
                for turn in transcript:
                    await add_message(
                        conn, self._conversation_id, turn["role"], turn["content"]
                    )

        # If we only have assistant text (no user STT), store as a single summary note
        full_text = self._transcript.get_full_text()
        if full_text and not transcript:
            async with pool.acquire() as conn:
                await add_message(
                    conn,
                    self._conversation_id,
                    "assistant",
                    full_text.strip(),
                )

        # Trigger async fact extraction
        asyncio.create_task(self._extract_facts_background(pool))

    async def _extract_facts_background(self, pool) -> None:
        try:
            async with pool.acquire() as conn:
                await process_conversation(
                    conn,
                    conversation_id=self._conversation_id,
                    user_id=self._user_id,
                    channel="voice",
                    llm_chat=self._llm_chat_wrapper,
                )
        except Exception:
            log.exception(
                "voice_fact_extraction_failed",
                conversation_id=str(self._conversation_id),
            )

    async def _llm_chat_wrapper(
        self, messages: list[dict], system_prompt: str = ""
    ) -> str:
        return await llm_client.chat(messages=messages, system_prompt=system_prompt)

    async def _cleanup(self) -> None:
        """Close WebSocket connections and HTTP session."""
        if self._pp_ws and not self._pp_ws.closed:
            await self._pp_ws.close()
        if self._session and not self._session.closed:
            await self._session.close()
