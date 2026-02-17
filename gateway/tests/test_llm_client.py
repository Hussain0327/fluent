"""Tests for LLM client — provider switching and failover."""

from unittest.mock import AsyncMock, patch

import pytest

from gateway.text import llm_client


@pytest.mark.asyncio
async def test_chat_claude_provider():
    """chat() should call Claude when provider is 'claude'."""
    with patch.object(llm_client, "_chat_claude", new_callable=AsyncMock) as mock:
        mock.return_value = "Hello from Claude"
        result = await llm_client.chat(
            messages=[{"role": "user", "content": "Hi"}],
            system_prompt="Be helpful",
            provider="claude",
            model="claude-sonnet-4-20250514",
        )
        assert result == "Hello from Claude"
        mock.assert_called_once()


@pytest.mark.asyncio
async def test_chat_openai_provider():
    """chat() should call OpenAI when provider is 'openai'."""
    with patch.object(llm_client, "_chat_openai", new_callable=AsyncMock) as mock:
        mock.return_value = "Hello from GPT"
        result = await llm_client.chat(
            messages=[{"role": "user", "content": "Hi"}],
            system_prompt="Be helpful",
            provider="openai",
            model="gpt-4o",
        )
        assert result == "Hello from GPT"
        mock.assert_called_once()


@pytest.mark.asyncio
async def test_chat_failover():
    """chat() should failover to secondary provider on error."""
    with (
        patch.object(
            llm_client, "_chat_claude", new_callable=AsyncMock, side_effect=Exception("API down")
        ),
        patch.object(
            llm_client, "_chat_openai", new_callable=AsyncMock, return_value="Fallback response"
        ) as mock_openai,
    ):
        result = await llm_client.chat(
            messages=[{"role": "user", "content": "Hi"}],
            system_prompt="Be helpful",
            provider="claude",
            model="claude-sonnet-4-20250514",
        )
        assert result == "Fallback response"
        mock_openai.assert_called_once()
