"""Unified LLM client supporting Claude and GPT-4o with automatic failover."""

from __future__ import annotations

import sys

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from gateway.config import settings
from gateway.utils.logging import get_logger

log = get_logger(__name__)

_anthropic: AsyncAnthropic | None = None
_openai: AsyncOpenAI | None = None


def _get_anthropic() -> AsyncAnthropic:
    global _anthropic
    if _anthropic is None:
        _anthropic = AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _anthropic


def _get_openai() -> AsyncOpenAI:
    global _openai
    if _openai is None:
        _openai = AsyncOpenAI(api_key=settings.openai_api_key)
    return _openai


async def _chat_claude(
    messages: list[dict],
    system_prompt: str,
    model: str,
) -> str:
    client = _get_anthropic()
    # Convert from OpenAI-style messages to Anthropic format
    anthropic_messages = []
    for msg in messages:
        if msg["role"] == "system":
            continue
        anthropic_messages.append({"role": msg["role"], "content": msg["content"]})

    resp = await client.messages.create(
        model=model,
        max_tokens=1024,
        system=system_prompt,
        messages=anthropic_messages,
    )
    return resp.content[0].text


async def _chat_openai(
    messages: list[dict],
    system_prompt: str,
    model: str,
) -> str:
    client = _get_openai()
    full_messages = [{"role": "system", "content": system_prompt}]
    full_messages.extend(messages)

    resp = await client.chat.completions.create(
        model=model,
        messages=full_messages,
        max_tokens=1024,
    )
    return resp.choices[0].message.content


# Failover mapping
_FAILOVER = {
    "claude": "openai",
    "openai": "claude",
}

# Default model per provider
_DEFAULT_MODELS = {
    "claude": "claude-sonnet-4-20250514",
    "openai": "gpt-4o",
}


def _get_chat_fn(provider: str):
    """Look up the chat function by provider name dynamically.

    This uses module-level attribute lookup so that unittest.mock.patch
    can intercept the function references.
    """
    _this = sys.modules[__name__]
    if provider == "claude":
        return getattr(_this, "_chat_claude")
    elif provider == "openai":
        return getattr(_this, "_chat_openai")
    raise ValueError(f"Unknown LLM provider: {provider}")


async def chat(
    messages: list[dict],
    system_prompt: str = "",
    provider: str | None = None,
    model: str | None = None,
) -> str:
    provider = provider or settings.default_llm_provider
    model = model or settings.default_llm_model

    chat_fn = _get_chat_fn(provider)
    try:
        return await chat_fn(messages, system_prompt, model)
    except Exception:
        fallback_provider = _FAILOVER.get(provider)
        if not fallback_provider:
            raise
        log.warning(
            "llm_failover",
            primary=provider,
            fallback=fallback_provider,
        )
        fallback_model = _DEFAULT_MODELS[fallback_provider]
        fallback_fn = _get_chat_fn(fallback_provider)
        return await fallback_fn(messages, system_prompt, fallback_model)
