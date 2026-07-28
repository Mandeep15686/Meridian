"""Base class for all HuggingFace Inference API model wrappers."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings

logger = logging.getLogger(__name__)

HF_INFERENCE_BASE = (
    os.getenv(
        "HF_INFERENCE_ENDPOINT",
        "https://router.huggingface.co",
    ).rstrip("/")
    + "/models"
)


class BaseHFModel(ABC):
    """
    Thin async wrapper around the HuggingFace Inference API.

    Subclasses implement ``_parse_response`` to convert the raw API
    response into the model's typed output format.

    Features:
    - Automatic retry with exponential backoff on 503 (model loading)
    - Optional Redis response caching
    - Structured logging with model_id and latency
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._url = f"{HF_INFERENCE_BASE}/{model_id}"
        self._headers = {
            "Authorization": f"Bearer {settings.HF_API_TOKEN}",
            "Content-Type": "application/json",
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _post(self, payload: dict[str, Any]) -> Any:
        """POST to the HF Inference API with retry on 503 (model loading)."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                self._url,
                headers=self._headers,
                json=payload,
            )

        if response.status_code == 503:
            # Model is loading — tenacity will retry
            raise httpx.HTTPStatusError(
                f"Model {self.model_id} is loading (503)",
                request=response.request,
                response=response,
            )

        if response.status_code != 200:
            raise httpx.HTTPStatusError(
                f"HF API error {response.status_code}: {response.text[:200]}",
                request=response.request,
                response=response,
            )

        return response.json()

    def _cache_key(self, payload: dict) -> str:
        """Generate a deterministic Redis cache key for a payload."""
        serialised = json.dumps(payload, sort_keys=True, default=str)
        digest = hashlib.sha256(serialised.encode()).hexdigest()[:16]
        return f"hf:{self.model_id.replace('/', ':')}:{digest}"

    async def _get_cached(self, key: str) -> Any | None:
        """Return cached result from Redis, or None on miss."""
        if settings.RETRIEVAL_CACHE_TTL <= 0:
            return None
        try:
            import redis.asyncio as aioredis
            r = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
            raw = await r.get(key)
            if raw:
                return json.loads(raw)
        except Exception as exc:
            logger.debug("Redis cache get failed: %s", exc)
        return None

    async def _set_cached(self, key: str, value: Any) -> None:
        """Store result in Redis with TTL."""
        if settings.RETRIEVAL_CACHE_TTL <= 0:
            return
        try:
            import redis.asyncio as aioredis
            r = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
            await r.setex(key, settings.RETRIEVAL_CACHE_TTL, json.dumps(value))
        except Exception as exc:
            logger.debug("Redis cache set failed: %s", exc)

    @abstractmethod
    def _parse_response(self, raw: Any) -> Any:
        """Convert raw API JSON into the typed output for this model."""
        ...

    async def predict(self, payload: dict[str, Any], use_cache: bool = True) -> Any:
        """Run inference, optionally using the Redis cache."""
        cache_key = self._cache_key(payload)

        if use_cache:
            cached = await self._get_cached(cache_key)
            if cached is not None:
                logger.debug("Cache hit for %s", self.model_id)
                return self._parse_response(cached)

        raw = await self._post(payload)
        result = self._parse_response(raw)

        if use_cache:
            await self._set_cached(cache_key, raw)

        return result

    async def health_check(self) -> bool:
        """Verify the model is reachable and loaded."""
        try:
            await self._post({"inputs": "test"})
            return True
        except Exception as exc:
            logger.warning("Health check failed for %s: %s", self.model_id, exc)
            return False
