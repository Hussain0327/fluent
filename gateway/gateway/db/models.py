"""Database operations using asyncpg directly.

All functions take a connection (or pool) as the first argument.
UUIDs are returned as strings.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

import asyncpg


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

async def get_or_create_user(
    conn: asyncpg.Connection | asyncpg.Pool,
    phone_number: str,
    display_name: str | None = None,
) -> dict[str, Any]:
    row = await conn.fetchrow(
        "SELECT * FROM users WHERE phone_number = $1", phone_number
    )
    if row:
        return dict(row)
    user_id = uuid.uuid4()
    row = await conn.fetchrow(
        """INSERT INTO users (id, phone_number, display_name)
           VALUES ($1, $2, $3)
           ON CONFLICT (phone_number) DO UPDATE SET phone_number = EXCLUDED.phone_number
           RETURNING *""",
        user_id,
        phone_number,
        display_name,
    )
    return dict(row)


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------

async def create_conversation(
    conn: asyncpg.Connection | asyncpg.Pool,
    user_id: uuid.UUID,
    channel: str,
    model_used: str | None = None,
) -> dict[str, Any]:
    conv_id = uuid.uuid4()
    row = await conn.fetchrow(
        """INSERT INTO conversations (id, user_id, channel, model_used)
           VALUES ($1, $2, $3, $4) RETURNING *""",
        conv_id,
        user_id,
        channel,
        model_used,
    )
    return dict(row)


async def end_conversation(
    conn: asyncpg.Connection | asyncpg.Pool,
    conversation_id: uuid.UUID,
    summary: str | None = None,
) -> None:
    await conn.execute(
        """UPDATE conversations SET ended_at = NOW(), summary = $2
           WHERE id = $1""",
        conversation_id,
        summary,
    )


async def get_latest_text_conversation(
    conn: asyncpg.Connection | asyncpg.Pool,
    user_id: uuid.UUID,
    idle_minutes: int = 30,
) -> dict[str, Any] | None:
    """Return the most recent text conversation if it's still within the idle window."""
    row = await conn.fetchrow(
        """SELECT c.* FROM conversations c
           JOIN messages m ON m.conversation_id = c.id
           WHERE c.user_id = $1
             AND c.channel = 'text'
             AND c.ended_at IS NULL
           GROUP BY c.id
           HAVING MAX(m.timestamp) > NOW() - INTERVAL '1 minute' * $2
           ORDER BY MAX(m.timestamp) DESC
           LIMIT 1""",
        user_id,
        idle_minutes,
    )
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

async def add_message(
    conn: asyncpg.Connection | asyncpg.Pool,
    conversation_id: uuid.UUID,
    role: str,
    content: str,
) -> dict[str, Any]:
    msg_id = uuid.uuid4()
    row = await conn.fetchrow(
        """INSERT INTO messages (id, conversation_id, role, content)
           VALUES ($1, $2, $3, $4) RETURNING *""",
        msg_id,
        conversation_id,
        role,
        content,
    )
    return dict(row)


async def get_conversation_messages(
    conn: asyncpg.Connection | asyncpg.Pool,
    conversation_id: uuid.UUID,
) -> list[dict[str, Any]]:
    rows = await conn.fetch(
        """SELECT * FROM messages WHERE conversation_id = $1
           ORDER BY timestamp ASC""",
        conversation_id,
    )
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Memories
# ---------------------------------------------------------------------------

async def store_memory(
    conn: asyncpg.Connection | asyncpg.Pool,
    user_id: uuid.UUID,
    memory_type: str,
    content: str,
    embedding: list[float],
    source_channel: str,
    source_conversation_id: uuid.UUID | None = None,
    confidence: float = 1.0,
    supersedes_id: uuid.UUID | None = None,
) -> dict[str, Any]:
    mem_id = uuid.uuid4()
    row = await conn.fetchrow(
        """INSERT INTO memories
           (id, user_id, type, content, embedding, confidence,
            source_channel, source_conversation_id, supersedes_id)
           VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) RETURNING *""",
        mem_id,
        user_id,
        memory_type,
        content,
        str(embedding),  # pgvector accepts text representation
        confidence,
        source_channel,
        source_conversation_id,
        supersedes_id,
    )
    return dict(row)


async def get_user_memories(
    conn: asyncpg.Connection | asyncpg.Pool,
    user_id: uuid.UUID,
    limit: int = 50,
) -> list[dict[str, Any]]:
    rows = await conn.fetch(
        """SELECT id, user_id, type, content, confidence,
                  source_channel, created_at
           FROM memories WHERE user_id = $1
           ORDER BY created_at DESC LIMIT $2""",
        user_id,
        limit,
    )
    return [dict(r) for r in rows]


async def search_memories_by_embedding(
    conn: asyncpg.Connection | asyncpg.Pool,
    user_id: uuid.UUID,
    query_embedding: list[float],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    rows = await conn.fetch(
        """SELECT id, user_id, type, content, confidence, source_channel, created_at,
                  1 - (embedding <=> $2) AS similarity
           FROM memories
           WHERE user_id = $1
             AND (expires_at IS NULL OR expires_at > NOW())
           ORDER BY embedding <=> $2
           LIMIT $3""",
        user_id,
        str(query_embedding),
        top_k,
    )
    return [dict(r) for r in rows]
