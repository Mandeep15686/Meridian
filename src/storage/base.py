"""Storage abstraction: local filesystem and GCP Cloud Storage backends."""

from __future__ import annotations

import logging
import shutil
from abc import ABC, abstractmethod
from typing import Any, cast
from pathlib import Path
from functools import lru_cache

from src.config import settings

logger = logging.getLogger(__name__)


def _as_bytes(value: Any) -> bytes:
    return cast(bytes, value)


def _as_bool(value: Any) -> bool:
    return cast(bool, value)


# ── Abstract base ─────────────────────────────────────────────────────────────


class StorageBackend(ABC):
    """Abstract interface for file storage operations."""

    @abstractmethod
    async def upload(
        self,
        data: bytes,
        key: str,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Upload bytes and return the storage key."""
        ...

    @abstractmethod
    async def download(self, key: str) -> bytes:
        """Download and return the bytes for a storage key."""
        ...

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete an object by key."""
        ...

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Return True if the key exists in storage."""
        ...

    @abstractmethod
    def public_url(self, key: str) -> str | None:
        """Return a public URL for the key, or None if not supported."""
        ...


# ── Local filesystem backend ──────────────────────────────────────────────────


class LocalStorageBackend(StorageBackend):
    """
    Local filesystem storage for development and testing.

    Files are stored under ``LOCAL_STORAGE_PATH/{uploads,reports}/``.
    """

    def __init__(self, root: Path | None = None) -> None:
        self._root = root or settings.LOCAL_STORAGE_PATH
        (self._root / "uploads").mkdir(parents=True, exist_ok=True)
        (self._root / "reports").mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        """Resolve a storage key to an absolute path, preventing traversal."""
        resolved = (self._root / key).resolve()
        if not str(resolved).startswith(str(self._root.resolve())):
            raise ValueError(f"Path traversal detected in storage key: {key}")
        return resolved

    async def upload(
        self,
        data: bytes,
        key: str,
        content_type: str = "application/octet-stream",
    ) -> str:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        logger.debug("LocalStorage: uploaded %s (%d bytes)", key, len(data))
        return key

    async def download(self, key: str) -> bytes:
        path = self._resolve(key)
        if not path.exists():
            raise FileNotFoundError(f"Storage key not found: {key}")
        return path.read_bytes()

    async def delete(self, key: str) -> None:
        path = self._resolve(key)
        if path.exists():
            path.unlink()
            logger.debug("LocalStorage: deleted %s", key)

    async def exists(self, key: str) -> bool:
        return self._resolve(key).exists()

    def public_url(self, key: str) -> str | None:
        return None  # No public URL for local storage


# ── GCP Cloud Storage backend ─────────────────────────────────────────────────


class GCSStorageBackend(StorageBackend):
    """
    GCP Cloud Storage backend for production deployments.

    Requires GCS_UPLOADS_BUCKET and GCS_REPORTS_BUCKET to be configured.
    Routing: keys starting with 'reports/' go to the reports bucket;
    everything else goes to the uploads bucket.
    """

    def __init__(self) -> None:
        from google.cloud import storage as gcs

        self._client = gcs.Client()
        self._uploads_bucket = self._client.bucket(settings.GCS_UPLOADS_BUCKET)
        self._reports_bucket = self._client.bucket(settings.GCS_REPORTS_BUCKET)

    def _bucket_and_key(self, key: str) -> tuple[Any, str]:
        if key.startswith("reports/"):
            return self._reports_bucket, key[len("reports/") :]
        return self._uploads_bucket, key

    async def upload(
        self,
        data: bytes,
        key: str,
        content_type: str = "application/octet-stream",
    ) -> str:
        import asyncio

        bucket, blob_name = self._bucket_and_key(key)
        blob = bucket.blob(blob_name)

        def _sync_upload() -> None:
            blob.upload_from_string(data, content_type=content_type)

        await asyncio.get_event_loop().run_in_executor(None, _sync_upload)
        logger.debug("GCS: uploaded %s (%d bytes)", key, len(data))
        return key

    async def download(self, key: str) -> bytes:
        import asyncio

        bucket, blob_name = self._bucket_and_key(key)
        blob = bucket.blob(blob_name)

        def _sync_download() -> bytes:
            return cast(bytes, blob.download_as_bytes())

        data: bytes = _as_bytes(await asyncio.to_thread(_sync_download))
        return data

    async def delete(self, key: str) -> None:
        import asyncio

        bucket, blob_name = self._bucket_and_key(key)
        blob = bucket.blob(blob_name)

        def _sync_delete() -> None:
            blob.delete()

        await asyncio.get_event_loop().run_in_executor(None, _sync_delete)

    async def exists(self, key: str) -> bool:
        import asyncio

        bucket, blob_name = self._bucket_and_key(key)
        blob = bucket.blob(blob_name)

        def _sync_exists() -> bool:
            return cast(bool, blob.exists())

        exists: bool = _as_bool(await asyncio.to_thread(_sync_exists))
        return exists

    def public_url(self, key: str) -> str | None:
        bucket, blob_name = self._bucket_and_key(key)
        return f"https://storage.googleapis.com/{bucket.name}/{blob_name}"


# ── Factory ───────────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_storage() -> StorageBackend:
    """Return the configured storage backend (singleton per process)."""
    if settings.STORAGE_BACKEND == "gcs":
        logger.info("Using GCS storage backend")
        return GCSStorageBackend()
    logger.info("Using local storage backend: %s", settings.LOCAL_STORAGE_PATH)
    return LocalStorageBackend()
