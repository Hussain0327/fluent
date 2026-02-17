"""SMS conversation grouping — groups messages into logical conversations."""

from __future__ import annotations

import uuid
from typing import Any

import asyncpg

from gateway.config import settings
from gateway.db.models import get_latest_text_conversation, get_conversation_messages
from gateway.memory.store import create_conversation


async def get_or_create_text_conversation(
    conn: asyncpg.Connection | asyncpg.Pool,
    user_id: uuid.UUID,
    model_used: str | None = None,
) -> dict[str, Any]:
    """Find the active text conversation or create a new one.

    A conversation is considered active if the last message was sent within
    the idle timeout window.
    """
    existing = await get_latest_text_conversation(
        conn, user_id, settings.conversation_idle_timeout_minutes
    )
    if existing:
        return existing
    return await create_conversation(conn, user_id, "text", model_used)


async def get_recent_messages(
    conn: asyncpg.Connection | asyncpg.Pool,
    conversation_id: uuid.UUID,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Get recent messages for context window building."""
    messages = await get_conversation_messages(conn, conversation_id)
    return messages[-limit:]
