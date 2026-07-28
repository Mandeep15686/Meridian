"""Storage abstraction — local filesystem and GCP Cloud Storage backends."""

from src.storage.base import (
    GCSStorageBackend,
    LocalStorageBackend,
    StorageBackend,
    get_storage,
)

__all__ = ["StorageBackend", "LocalStorageBackend", "GCSStorageBackend", "get_storage"]
