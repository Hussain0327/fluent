"""Twilio request signature validation."""

from __future__ import annotations

from aiohttp import web
from twilio.request_validator import RequestValidator

from gateway.config import settings
from gateway.utils.logging import get_logger

log = get_logger(__name__)

_validator: RequestValidator | None = None


def _get_validator() -> RequestValidator:
    global _validator
    if _validator is None:
        _validator = RequestValidator(settings.twilio_auth_token)
    return _validator


def validate_twilio_request(request: web.Request, params: dict) -> bool:
    """Validate that the request actually came from Twilio."""
    if not settings.twilio_auth_token:
        # Skip validation if no auth token configured (development mode)
        return True

    validator = _get_validator()
    signature = request.headers.get("X-Twilio-Signature", "")
    url = str(request.url)

    is_valid = validator.validate(url, params, signature)
    if not is_valid:
        log.warning("twilio_signature_invalid", url=url)
    return is_valid
