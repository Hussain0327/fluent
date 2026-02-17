"""Twilio webhook handlers — incoming voice calls, SMS, and WebSocket streams."""

from __future__ import annotations

from aiohttp import web

from gateway.db.connection import get_pool
from gateway.telephony.twiml import sms_response, voice_stream_response
from gateway.telephony.validation import validate_twilio_request
from gateway.text.handler import handle_sms
from gateway.utils.logging import get_logger
from gateway.voice.bridge import VoiceBridge

log = get_logger(__name__)


async def handle_voice_incoming(request: web.Request) -> web.Response:
    """POST /voice/incoming — Twilio calls this when a voice call arrives.

    Returns TwiML that connects the call to our WebSocket stream endpoint.
    """
    data = await request.post()
    params = {k: v for k, v in data.items()}

    if not validate_twilio_request(request, params):
        return web.Response(status=403, text="Invalid signature")

    caller = params.get("From", "unknown")
    log.info("voice_call_incoming", caller=caller)

    # Build the WebSocket URL for Twilio Media Streams
    # Twilio connects to this URL to stream audio
    scheme = "wss" if request.secure else "ws"
    host = request.host
    stream_url = f"{scheme}://{host}/voice/stream?caller={caller}"

    twiml = voice_stream_response(stream_url)
    return web.Response(text=twiml, content_type="application/xml")


async def handle_sms_incoming(request: web.Request) -> web.Response:
    """POST /sms/incoming — Twilio calls this when an SMS arrives."""
    data = await request.post()
    params = {k: v for k, v in data.items()}

    if not validate_twilio_request(request, params):
        return web.Response(status=403, text="Invalid signature")

    from_number = params.get("From", "")
    body = params.get("Body", "")

    log.info("sms_incoming", from_number=from_number, body_len=len(body))

    pool = await get_pool()
    response_text = await handle_sms(pool, from_number, body)

    twiml = sms_response(response_text)
    return web.Response(text=twiml, content_type="application/xml")


async def handle_voice_stream(request: web.Request) -> web.WebSocketResponse:
    """WS /voice/stream — Twilio Media Streams WebSocket endpoint.

    On connection: create VoiceBridge and start bridging audio.
    On disconnect: clean up.
    """
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    caller = request.query.get("caller", "unknown")
    log.info("voice_stream_connected", caller=caller)

    bridge = VoiceBridge()
    try:
        await bridge.start(twilio_ws=ws, caller_phone=caller)
    except Exception:
        log.exception("voice_bridge_error", caller=caller)
    finally:
        log.info("voice_stream_disconnected", caller=caller)

    return ws
