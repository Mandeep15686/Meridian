"""Sentence similarity model — re-exported from retrieval module for clean imports."""

# The gate node imports SimilarityModel from this module.
# The actual implementation lives in retrieval.py alongside the cross-encoder reranker
# since both are retrieval-quality models loaded in-process.

from src.models.retrieval import SimilarityModel, CrossEncoderReranker

__all__ = ["SimilarityModel", "CrossEncoderReranker"]
