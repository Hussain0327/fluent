"""Audio format conversion between Twilio (mulaw 8kHz) and PersonaPlex (Opus 24kHz).

Twilio Media Streams sends/receives base64-encoded mulaw (G.711) at 8kHz mono.
PersonaPlex expects/produces Opus frames at 24kHz mono.

Pipeline:
  Twilio -> mulaw 8kHz -> PCM float32 8kHz -> resample 24kHz -> Opus encode -> PersonaPlex
  PersonaPlex -> Opus decode -> PCM float32 24kHz -> resample 8kHz -> mulaw encode -> Twilio
"""

from __future__ import annotations

import numpy as np
import soxr

TWILIO_RATE = 8000
PERSONAPLEX_RATE = 24000

# mulaw bias constant (ITU-T G.711)
_MULAW_BIAS = 0x84
_MULAW_CLIP = 32635

# Precompute mulaw decode table
_MULAW_DECODE_TABLE = np.zeros(256, dtype=np.float32)
for _i in range(256):
    _val = ~_i & 0xFF
    _sign = _val & 0x80
    _exponent = (_val >> 4) & 0x07
    _mantissa = _val & 0x0F
    _sample = ((_mantissa << 3) + _MULAW_BIAS) << _exponent
    _sample -= _MULAW_BIAS
    if _sign:
        _sample = -_sample
    _MULAW_DECODE_TABLE[_i] = _sample / 32768.0


def mulaw_to_pcm(data: bytes) -> np.ndarray:
    """Decode mulaw bytes to float32 PCM in [-1, 1]."""
    indices = np.frombuffer(data, dtype=np.uint8)
    return _MULAW_DECODE_TABLE[indices].copy()


def _encode_mulaw_sample(sample: int) -> int:
    """Encode a single signed 16-bit PCM sample to a mulaw byte."""
    sign = 0
    if sample < 0:
        sign = 0x80
        sample = -sample
    if sample > _MULAW_CLIP:
        sample = _MULAW_CLIP
    sample += _MULAW_BIAS

    exponent = 7
    mask = 0x4000
    for e in range(7, 0, -1):
        if sample & mask:
            exponent = e
            break
        mask >>= 1
    else:
        exponent = 0

    mantissa = (sample >> (exponent + 3)) & 0x0F
    return ~(sign | (exponent << 4) | mantissa) & 0xFF


# Precompute encode table for all 65536 uint16 values
# (interpreted as int16 for the encoding)
_MULAW_ENCODE_TABLE = np.empty(65536, dtype=np.uint8)
for _u in range(65536):
    # Interpret as signed int16
    _s = _u if _u < 32768 else _u - 65536
    _MULAW_ENCODE_TABLE[_u] = _encode_mulaw_sample(_s)


def pcm_to_mulaw(pcm: np.ndarray) -> bytes:
    """Encode float32 PCM [-1, 1] to mulaw bytes."""
    pcm_clipped = np.clip(pcm, -1.0, 1.0)
    int16_data = (pcm_clipped * 32767).astype(np.int16)
    uint16_view = int16_data.view(np.uint16)
    return _MULAW_ENCODE_TABLE[uint16_view].tobytes()


def resample(pcm: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Resample float32 PCM from one sample rate to another."""
    if from_rate == to_rate:
        return pcm
    return soxr.resample(pcm, from_rate, to_rate)


def mulaw_8k_to_pcm_24k(data: bytes) -> np.ndarray:
    """Full pipeline: mulaw 8kHz -> float32 PCM 24kHz."""
    pcm_8k = mulaw_to_pcm(data)
    return resample(pcm_8k, TWILIO_RATE, PERSONAPLEX_RATE)


def pcm_24k_to_mulaw_8k(pcm_24k: np.ndarray) -> bytes:
    """Full pipeline: float32 PCM 24kHz -> mulaw 8kHz."""
    pcm_8k = resample(pcm_24k, PERSONAPLEX_RATE, TWILIO_RATE)
    return pcm_to_mulaw(pcm_8k)
