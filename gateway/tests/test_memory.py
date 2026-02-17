"""Tests for memory layer — CRUD, semantic search, fact versioning."""

import uuid

import pytest

from gateway.memory.retrieval import format_memories_for_prompt


def test_format_memories_empty():
    assert format_memories_for_prompt([]) == ""


def test_format_memories_with_facts():
    memories = [
        {"type": "fact", "content": "User's name is Alice"},
        {"type": "preference", "content": "User prefers dark mode"},
    ]
    result = format_memories_for_prompt(memories)
    assert "<memories>" in result
    assert "[fact] User's name is Alice" in result
    assert "[preference] User prefers dark mode" in result
    assert "</memories>" in result


def test_format_memories_missing_type():
    """Should default to 'fact' if type is missing."""
    memories = [{"content": "Something important"}]
    result = format_memories_for_prompt(memories)
    assert "[fact] Something important" in result
