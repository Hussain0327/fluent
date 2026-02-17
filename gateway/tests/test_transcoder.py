"""Tests for audio transcoder — mulaw <-> PCM <-> Opus round-trip."""

import numpy as np
import pytest

from gateway.voice.transcoder import (
    PERSONAPLEX_RATE,
    TWILIO_RATE,
    mulaw_8k_to_pcm_24k,
    mulaw_to_pcm,
    pcm_24k_to_mulaw_8k,
    pcm_to_mulaw,
    resample,
)


def test_mulaw_decode_range():
    """Decoded mulaw values should be in [-1, 1]."""
    all_bytes = bytes(range(256))
    pcm = mulaw_to_pcm(all_bytes)
    assert pcm.dtype == np.float32
    assert pcm.min() >= -1.0
    assert pcm.max() <= 1.0


def test_mulaw_roundtrip():
    """mulaw -> PCM -> mulaw should approximately preserve the signal."""
    # Generate some mulaw test data (silence-ish)
    original = bytes([0xFF] * 160)  # 20ms at 8kHz
    pcm = mulaw_to_pcm(original)
    reencoded = pcm_to_mulaw(pcm)
    # Re-decode and compare
    pcm2 = mulaw_to_pcm(reencoded)
    # Should be close (mulaw is lossy but self-consistent)
    np.testing.assert_allclose(pcm, pcm2, atol=0.01)


def test_resample_identity():
    """Resampling to the same rate should return the same data."""
    pcm = np.random.randn(480).astype(np.float32) * 0.5
    result = resample(pcm, 24000, 24000)
    np.testing.assert_array_equal(pcm, result)


def test_resample_ratio():
    """Resampling should produce the correct number of samples."""
    pcm_8k = np.random.randn(160).astype(np.float32) * 0.5  # 20ms at 8kHz
    pcm_24k = resample(pcm_8k, TWILIO_RATE, PERSONAPLEX_RATE)
    expected_samples = 160 * PERSONAPLEX_RATE // TWILIO_RATE  # 480
    assert len(pcm_24k) == expected_samples


def test_full_pipeline_roundtrip():
    """mulaw 8kHz -> PCM 24kHz -> mulaw 8kHz should approximately preserve length."""
    mulaw_data = bytes([0x80 + i % 64 for i in range(160)])  # 20ms at 8kHz
    pcm_24k = mulaw_8k_to_pcm_24k(mulaw_data)
    assert len(pcm_24k) == 480  # 20ms at 24kHz

    mulaw_back = pcm_24k_to_mulaw_8k(pcm_24k)
    assert len(mulaw_back) == 160  # 20ms at 8kHz


def test_sine_wave_preservation():
    """A 440Hz sine wave should survive the transcode pipeline."""
    # Generate 20ms of 440Hz sine at 8kHz as mulaw
    t = np.arange(160) / TWILIO_RATE
    sine_8k = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5
    mulaw = pcm_to_mulaw(sine_8k)

    # Full round-trip
    pcm_24k = mulaw_8k_to_pcm_24k(mulaw)
    mulaw_back = pcm_24k_to_mulaw_8k(pcm_24k)
    pcm_back = mulaw_to_pcm(mulaw_back)

    # The signal should still be roughly sinusoidal
    # Check that peak-to-peak is preserved (within mulaw quantization error)
    assert pcm_back.max() > 0.3
    assert pcm_back.min() < -0.3
