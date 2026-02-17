from openai import AsyncOpenAI

from gateway.config import settings
from gateway.utils.logging import get_logger

log = get_logger(__name__)

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


async def embed_text(text: str) -> list[float]:
    client = _get_client()
    resp = await client.embeddings.create(
        input=text,
        model=settings.embedding_model,
    )
    return resp.data[0].embedding


async def embed_batch(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    client = _get_client()
    resp = await client.embeddings.create(
        input=texts,
        model=settings.embedding_model,
    )
    return [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]
