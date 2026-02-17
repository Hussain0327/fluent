"""Semantic memory retrieval using pgvector."""

from __future__ import annotations

import uuid
from typing import Any

import asyncpg

from gateway.config import settings
from gateway.db.models import search_memories_by_embedding
from gateway.memory.embeddings import embed_text
from gateway.utils.logging import get_logger

log = get_logger(__name__)


async def retrieve_relevant_memories(
    conn: asyncpg.Connection | asyncpg.Pool,
    user_id: uuid.UUID,
    query_text: str,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    top_k = top_k or settings.memory_top_k
    query_embedding = await embed_text(query_text)
    memories = await search_memories_by_embedding(
        conn, user_id, query_embedding, top_k
    )
    log.info(
        "memories_retrieved",
        user_id=str(user_id),
        query_preview=query_text[:80],
        count=len(memories),
    )
    return memories


def format_memories_for_prompt(memories: list[dict[str, Any]]) -> str:
    if not memories:
        return ""
    lines = ["<memories>"]
    for mem in memories:
        mtype = mem.get("type", "fact")
        content = mem["content"]
        lines.append(f"- [{mtype}] {content}")
    lines.append("</memories>")
    return "\n".join(lines)


async def build_memory_context(
    conn: asyncpg.Connection | asyncpg.Pool,
    user_id: uuid.UUID,
    query_text: str,
) -> str:
    memories = await retrieve_relevant_memories(conn, user_id, query_text)
    return format_memories_for_prompt(memories)
