"""Unit tests for the hybrid retrieval pipeline components."""

from __future__ import annotations

import pytest

from src.rag.retrieve import reciprocal_rank_fusion
from tests.conftest import make_retrieved_chunk


class TestHybridRetrieval:
    """Tests for individual retrieval stages (mocking DB calls)."""

    def test_rrf_merges_disjoint_lists(self):
        dense = [
            {"chunk_id": "A", "content": "A", "dense_score": 0.9, "dense_rank": 1},
            {"chunk_id": "B", "content": "B", "dense_score": 0.8, "dense_rank": 2},
        ]
        bm25 = [
            {"chunk_id": "C", "content": "C", "bm25_score": 10.0, "bm25_rank": 1},
            {"chunk_id": "A", "content": "A", "bm25_score": 8.0, "bm25_rank": 2},
        ]
        fused = reciprocal_rank_fusion(dense, bm25)
        chunk_ids = [r["chunk_id"] for r in fused]
        assert "A" in chunk_ids
        assert "B" in chunk_ids
        assert "C" in chunk_ids

    def test_rrf_dual_list_chunk_outranks_single(self):
        """A chunk in both lists should outrank one appearing in only one."""
        dense = [
            {"chunk_id": "BOTH", "content": "x", "dense_score": 0.7, "dense_rank": 1},
            {
                "chunk_id": "DENSE_ONLY",
                "content": "y",
                "dense_score": 0.9,
                "dense_rank": 2,
            },
        ]
        bm25 = [
            {"chunk_id": "BOTH", "content": "x", "bm25_score": 5.0, "bm25_rank": 1},
        ]
        fused = reciprocal_rank_fusion(dense, bm25)
        ids = [r["chunk_id"] for r in fused]
        both_pos = ids.index("BOTH")
        dense_only_pos = ids.index("DENSE_ONLY")
        assert both_pos < dense_only_pos

    def test_rrf_empty_inputs_return_empty(self):
        assert reciprocal_rank_fusion([], []) == []

    def test_rrf_single_list_returns_all(self):
        dense = [
            {"chunk_id": "A", "content": "A", "dense_score": 0.9, "dense_rank": 1},
            {"chunk_id": "B", "content": "B", "dense_score": 0.8, "dense_rank": 2},
        ]
        fused = reciprocal_rank_fusion(dense, [])
        assert len(fused) == 2

    def test_rrf_score_positive_for_all_chunks(self):
        dense = [{"chunk_id": "X", "content": "x", "dense_score": 0.5, "dense_rank": 1}]
        bm25 = [{"chunk_id": "Y", "content": "y", "bm25_score": 3.0, "bm25_rank": 1}]
        fused = reciprocal_rank_fusion(dense, bm25)
        for r in fused:
            assert r["rrf_score"] > 0

    @pytest.mark.parametrize("k", [1, 10, 60, 100])
    def test_rrf_different_k_values_produce_valid_scores(self, k: int):
        dense = [{"chunk_id": "A", "content": "A", "dense_score": 0.9, "dense_rank": 1}]
        bm25 = [{"chunk_id": "A", "content": "A", "bm25_score": 5.0, "bm25_rank": 1}]
        fused = reciprocal_rank_fusion(dense, bm25, k=k)
        assert len(fused) == 1
        expected_score = 2.0 / (k + 1)  # rank 1 in both lists → 1/(k+1) + 1/(k+1)
        assert abs(fused[0]["rrf_score"] - expected_score) < 1e-9


class TestRetrievedChunkState:
    """Tests for the RetrievedChunk dataclass used in state."""

    def test_make_retrieved_chunk_defaults(self):
        chunk = make_retrieved_chunk()
        assert chunk.chunk_id == "chunk-gdpr-001"
        assert chunk.regulation == "gdpr"
        assert chunk.final_rank == 1
        assert chunk.rerank_score > 0

    def test_make_retrieved_chunk_custom(self):
        chunk = make_retrieved_chunk(
            chunk_id="custom-001",
            regulation="soc2",
            article="CC6.1",
            final_rank=3,
        )
        assert chunk.chunk_id == "custom-001"
        assert chunk.regulation == "soc2"
        assert chunk.article == "CC6.1"
        assert chunk.final_rank == 3

    def test_retrieved_chunk_content_default(self):
        chunk = make_retrieved_chunk(regulation="soc2", article="CC6.1")
        assert "SOC2" in chunk.content.upper() or "soc2" in chunk.content.lower()

    def test_retrieved_chunk_with_custom_content(self):
        custom_content = "The period for which personal data will be stored..."
        chunk = make_retrieved_chunk(content=custom_content)
        assert chunk.content == custom_content
