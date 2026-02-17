"""High-level memory operations built on top of db.models."""

from __future__ import annotations

import uuid
from typing import Any

import asyncpg

from gateway.db import models
from gateway.memory.embeddings import embed_text
from gateway.utils.logging import get_logger

log = get_logger(__name__)


async def get_or_create_user(
    conn: asyncpg.Connection | asyncpg.Pool,
    phone_number: str,
) -> dict[str, Any]:
    return await models.get_or_create_user(conn, phone_number)


async def create_conversation(
    conn: asyncpg.Connection | asyncpg.Pool,
    user_id: uuid.UUID,
    channel: str,
    model_used: str | None = None,
) -> dict[str, Any]:
    conv = await models.create_conversation(conn, user_id, channel, model_used)
    log.info(
        "conversation_created",
        conversation_id=str(conv["id"]),
        channel=channel,
    )
    return conv


async def end_conversation(
    conn: asyncpg.Connection | asyncpg.Pool,
    conversation_id: uuid.UUID,
    summary: str | None = None,
) -> None:
    await models.end_conversation(conn, conversation_id, summary)
    log.info("conversation_ended", conversation_id=str(conversation_id))


async def add_message(
    conn: asyncpg.Connection | asyncpg.Pool,
    conversation_id: uuid.UUID,
    role: str,
    content: str,
) -> dict[str, Any]:
    return await models.add_message(conn, conversation_id, role, content)


async def store_memory(
    conn: asyncpg.Connection | asyncpg.Pool,
    user_id: uuid.UUID,
    memory_type: str,
    content: str,
    source_channel: str,
    source_conversation_id: uuid.UUID | None = None,
    confidence: float = 1.0,
    supersedes_id: uuid.UUID | None = None,
) -> dict[str, Any]:
    embedding = await embed_text(content)
    mem = await models.store_memory(
        conn,
        user_id=user_id,
        memory_type=memory_type,
        content=content,
        embedding=embedding,
        source_channel=source_channel,
        source_conversation_id=source_conversation_id,
        confidence=confidence,
        supersedes_id=supersedes_id,
    )
    log.info(
        "memory_stored",
        memory_id=str(mem["id"]),
        memory_type=memory_type,
        user_id=str(user_id),
    )
    return mem


async def get_user_memories(
    conn: asyncpg.Connection | asyncpg.Pool,
    user_id: uuid.UUID,
    limit: int = 50,
) -> list[dict[str, Any]]:
    return await models.get_user_memories(conn, user_id, limit)
