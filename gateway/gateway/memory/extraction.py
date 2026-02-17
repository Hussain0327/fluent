"""Post-conversation fact extraction via LLM."""

from __future__ import annotations

import json
import uuid

import asyncpg

from gateway.db.models import get_conversation_messages
from gateway.memory.embeddings import embed_batch
from gateway.memory.store import store_memory, end_conversation
from gateway.utils.logging import get_logger

log = get_logger(__name__)

FACT_EXTRACTION_PROMPT = """\
You are a memory extraction system. Analyze the following conversation and extract \
discrete facts about the user. Return a JSON array of objects, each with:
- "type": one of "fact", "preference", "action_item"
- "content": a concise statement of the fact (always from user's perspective, e.g. "User's name is Alice")
- "confidence": float 0.0–1.0 indicating how certain this fact is

Only extract facts explicitly stated or strongly implied by the user. Do not infer or speculate.
Return ONLY the JSON array, no other text.

Conversation:
{transcript}"""

SUMMARY_PROMPT = """\
Write a one-paragraph summary of this conversation. Focus on key topics discussed, \
decisions made, and any commitments. Be concise.

Conversation:
{transcript}"""


def _build_transcript(messages: list[dict]) -> str:
    lines = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


async def extract_facts(
    llm_chat,  # callable: async (messages, system) -> str
    transcript: str,
) -> list[dict]:
    prompt = FACT_EXTRACTION_PROMPT.format(transcript=transcript)
    response = await llm_chat(
        messages=[{"role": "user", "content": prompt}],
        system_prompt="You are a precise fact extraction system. Return only valid JSON.",
    )
    try:
        facts = json.loads(response)
        if not isinstance(facts, list):
            facts = []
    except json.JSONDecodeError:
        log.warning("fact_extraction_json_parse_failed", response=response[:200])
        facts = []
    return facts


async def extract_summary(
    llm_chat,  # callable: async (messages, system) -> str
    transcript: str,
) -> str:
    prompt = SUMMARY_PROMPT.format(transcript=transcript)
    response = await llm_chat(
        messages=[{"role": "user", "content": prompt}],
        system_prompt="You are a conversation summarizer.",
    )
    return response.strip()


async def process_conversation(
    conn: asyncpg.Connection | asyncpg.Pool,
    conversation_id: uuid.UUID,
    user_id: uuid.UUID,
    channel: str,
    llm_chat,  # callable
) -> None:
    """Run post-conversation processing: extract facts + summary, store memories."""
    messages = await get_conversation_messages(conn, conversation_id)
    if not messages:
        return

    transcript = _build_transcript(messages)

    facts = await extract_facts(llm_chat, transcript)
    summary = await extract_summary(llm_chat, transcript)

    # Store extracted facts as memories
    for fact in facts:
        content = fact.get("content", "")
        if not content:
            continue
        await store_memory(
            conn,
            user_id=user_id,
            memory_type=fact.get("type", "fact"),
            content=content,
            source_channel=channel,
            source_conversation_id=conversation_id,
            confidence=fact.get("confidence", 1.0),
        )

    # Store conversation summary as a memory
    if summary:
        await store_memory(
            conn,
            user_id=user_id,
            memory_type="summary",
            content=summary,
            source_channel=channel,
            source_conversation_id=conversation_id,
        )

    # Update conversation with summary
    await end_conversation(conn, conversation_id, summary)

    log.info(
        "conversation_processed",
        conversation_id=str(conversation_id),
        facts_count=len(facts),
        has_summary=bool(summary),
    )
