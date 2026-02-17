"""TwiML response builders for voice and SMS."""

from __future__ import annotations

from twilio.twiml.messaging_response import MessagingResponse
from twilio.twiml.voice_response import Connect, VoiceResponse


def voice_stream_response(stream_url: str) -> str:
    """Build TwiML to connect a voice call to a WebSocket stream."""
    response = VoiceResponse()
    connect = Connect()
    connect.stream(url=stream_url)
    response.append(connect)
    return str(response)


def sms_response(body: str) -> str:
    """Build TwiML to respond to an SMS."""
    response = MessagingResponse()
    response.message(body)
    return str(response)
