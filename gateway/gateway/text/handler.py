"""SMS message handler — receives SMS, builds context, calls LLM, responds."""

from __future__ import annotations

import asyncio

import asyncpg

from gateway.config import settings
from gateway.db.models import get_or_create_user
from gateway.memory.extraction import process_conversation
from gateway.memory.retrieval import build_memory_context
from gateway.memory.store import add_message
from gateway.text import llm_client
from gateway.text.conversation import get_or_create_text_conversation, get_recent_messages
from gateway.utils.logging import get_logger
from gateway.utils.phone import normalize_e164

log = get_logger(__name__)

SYSTEM_PROMPT_TEMPLATE = """\
You are a helpful, friendly AI assistant communicating via text message. \
Keep responses concise and natural for SMS — avoid overly long messages.

{memory_context}

Respond naturally to the user's message."""


async def handle_sms(
    pool: asyncpg.Pool,
    from_number: str,
    body: str,
) -> str:
    """Process an incoming SMS and return the response text."""
    from_number = normalize_e164(from_number)

    async with pool.acquire() as conn:
        # 1. Get or create user
        user = await get_or_create_user(conn, from_number)
        user_id = user["id"]

        # 2. Get or create active conversation
        conversation = await get_or_create_text_conversation(
            conn, user_id, settings.default_llm_model
        )
        conversation_id = conversation["id"]

        # 3. Store user message
        await add_message(conn, conversation_id, "user", body)

        # 4. Retrieve relevant memories
        memory_context = await build_memory_context(conn, user_id, body)

        # 5. Build messages array with conversation history
        recent = await get_recent_messages(conn, conversation_id, limit=20)
        messages = []
        for msg in recent:
            if msg["role"] in ("user", "assistant"):
                messages.append({"role": msg["role"], "content": msg["content"]})

        # 6. Build system prompt with memory
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            memory_context=memory_context,
        )

        # 7. Call LLM
        response = await llm_client.chat(
            messages=messages,
            system_prompt=system_prompt,
        )

        # 8. Store assistant response
        await add_message(conn, conversation_id, "assistant", response)

        log.info(
            "sms_handled",
            user_id=str(user_id),
            conversation_id=str(conversation_id),
            body_len=len(body),
            response_len=len(response),
        )

    # 9. Async: run fact extraction (fire-and-forget)
    asyncio.create_task(
        _extract_facts_background(pool, conversation_id, user_id)
    )

    return response


async def _extract_facts_background(
    pool: asyncpg.Pool,
    conversation_id,
    user_id,
) -> None:
    try:
        async with pool.acquire() as conn:
            await process_conversation(
                conn,
                conversation_id=conversation_id,
                user_id=user_id,
                channel="text",
                llm_chat=_llm_chat_wrapper,
            )
    except Exception:
        log.exception("fact_extraction_failed", conversation_id=str(conversation_id))


async def _llm_chat_wrapper(messages: list[dict], system_prompt: str = "") -> str:
    return await llm_client.chat(messages=messages, system_prompt=system_prompt)
