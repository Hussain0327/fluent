"""Tests for fact extraction from sample transcripts."""

import json

import pytest

from gateway.memory.extraction import FACT_EXTRACTION_PROMPT, SUMMARY_PROMPT, extract_facts, extract_summary


@pytest.mark.asyncio
async def test_extract_facts_valid_json():
    """extract_facts should parse valid JSON from LLM response."""
    fake_facts = [
        {"type": "fact", "content": "User's name is Bob", "confidence": 0.95},
        {"type": "preference", "content": "User likes coffee", "confidence": 0.8},
    ]

    async def mock_llm(messages, system_prompt=""):
        return json.dumps(fake_facts)

    result = await extract_facts(mock_llm, "user: My name is Bob and I like coffee")
    assert len(result) == 2
    assert result[0]["content"] == "User's name is Bob"
    assert result[1]["type"] == "preference"


@pytest.mark.asyncio
async def test_extract_facts_invalid_json():
    """extract_facts should return empty list on invalid JSON."""
    async def mock_llm(messages, system_prompt=""):
        return "This is not JSON at all"

    result = await extract_facts(mock_llm, "some transcript")
    assert result == []


@pytest.mark.asyncio
async def test_extract_facts_non_list():
    """extract_facts should return empty list if LLM returns non-list JSON."""
    async def mock_llm(messages, system_prompt=""):
        return '{"single": "object"}'

    result = await extract_facts(mock_llm, "some transcript")
    assert result == []


@pytest.mark.asyncio
async def test_extract_summary():
    """extract_summary should return trimmed text from LLM."""
    async def mock_llm(messages, system_prompt=""):
        return "  The user discussed their travel plans to Japan.  "

    result = await extract_summary(mock_llm, "some transcript")
    assert result == "The user discussed their travel plans to Japan."


def test_fact_extraction_prompt_template():
    """The prompt template should accept a transcript variable."""
    filled = FACT_EXTRACTION_PROMPT.format(transcript="Hello world")
    assert "Hello world" in filled
    assert "JSON array" in filled


def test_summary_prompt_template():
    filled = SUMMARY_PROMPT.format(transcript="Test conversation")
    assert "Test conversation" in filled
