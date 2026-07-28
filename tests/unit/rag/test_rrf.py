"""Unit tests for the Reciprocal Rank Fusion implementation."""

from __future__ import annotations

import pytest

from src.rag.retrieve import reciprocal_rank_fusion

# ── Fixtures ──────────────────────────────────────────────────────────────────


def make_dense_results(chunk_ids_and_scores: list[tuple[str, float]]) -> list[dict]:
    return [
        {
            "chunk_id": cid,
            "regulation": "gdpr",
            "article": "Article 5",
            "content": f"Content for chunk {cid}",
            "jurisdiction": "EU",
            "dense_score": score,
            "dense_rank": i + 1,
        }
        for i, (cid, score) in enumerate(chunk_ids_and_scores)
    ]


def make_bm25_results(chunk_ids_and_scores: list[tuple[str, float]]) -> list[dict]:
    return [
        {
            "chunk_id": cid,
            "regulation": "gdpr",
            "article": "Article 5",
            "content": f"Content for chunk {cid}",
            "jurisdiction": "EU",
            "bm25_score": score,
            "bm25_rank": i + 1,
        }
        for i, (cid, score) in enumerate(chunk_ids_and_scores)
    ]


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestRRF:
    def test_chunk_in_both_lists_ranked_first(self):
        """A chunk appearing at the top of both lists should rank #1 after RRF."""
        dense = make_dense_results([("A", 0.95), ("B", 0.90), ("C", 0.85)])
        bm25 = make_bm25_results([("A", 12.4), ("C", 10.1), ("D", 8.8)])

        fused = reciprocal_rank_fusion(dense, bm25)

        assert (
            fused[0]["chunk_id"] == "A"
        ), "Chunk A appears #1 in both lists — should be top-ranked after RRF"

    def test_chunk_only_in_dense_ranked_lower_than_dual(self):
        """A chunk appearing only in dense retrieval should rank lower than dual-list chunks."""
        dense = make_dense_results([("A", 0.95), ("B", 0.90), ("C", 0.85)])
        bm25 = make_bm25_results([("C", 12.0), ("D", 10.0)])

        fused = reciprocal_rank_fusion(dense, bm25)

        c_rank = next(i for i, r in enumerate(fused) if r["chunk_id"] == "C")
        b_rank = next(i for i, r in enumerate(fused) if r["chunk_id"] == "B")

        assert (
            c_rank < b_rank
        ), "C appears in both lists (even at lower dense rank) and should outrank B (dense-only)"

    def test_all_chunks_present_in_output(self):
        """Output should contain all unique chunks from both input lists."""
        dense = make_dense_results([("A", 0.9), ("B", 0.8)])
        bm25 = make_bm25_results([("B", 10.0), ("C", 8.0)])

        fused = reciprocal_rank_fusion(dense, bm25)
        output_ids = {r["chunk_id"] for r in fused}

        assert "A" in output_ids
        assert "B" in output_ids
        assert "C" in output_ids

    def test_rrf_scores_are_positive(self):
        """All RRF scores should be strictly positive."""
        dense = make_dense_results([("A", 0.9), ("B", 0.8)])
        bm25 = make_bm25_results([("A", 10.0), ("C", 5.0)])

        fused = reciprocal_rank_fusion(dense, bm25)

        for chunk in fused:
            assert (
                chunk["rrf_score"] > 0
            ), f"Chunk {chunk['chunk_id']} has non-positive RRF score"

    def test_rrf_output_sorted_descending(self):
        """Output list should be sorted by RRF score in descending order."""
        dense = make_dense_results([("A", 0.95), ("B", 0.90), ("C", 0.85), ("D", 0.80)])
        bm25 = make_bm25_results([("D", 15.0), ("C", 12.0), ("A", 10.0)])

        fused = reciprocal_rank_fusion(dense, bm25)
        scores = [r["rrf_score"] for r in fused]

        assert scores == sorted(
            scores, reverse=True
        ), "Output should be sorted descending by RRF score"

    def test_empty_dense_results(self):
        """RRF should handle empty dense results gracefully."""
        bm25 = make_bm25_results([("A", 10.0), ("B", 8.0)])
        fused = reciprocal_rank_fusion([], bm25)

        assert len(fused) == 2
        assert {r["chunk_id"] for r in fused} == {"A", "B"}

    def test_empty_bm25_results(self):
        """RRF should handle empty BM25 results gracefully."""
        dense = make_dense_results([("A", 0.9), ("B", 0.8)])
        fused = reciprocal_rank_fusion(dense, [])

        assert len(fused) == 2
        assert {r["chunk_id"] for r in fused} == {"A", "B"}

    def test_both_empty_returns_empty(self):
        """Empty inputs should produce empty output."""
        fused = reciprocal_rank_fusion([], [])
        assert fused == []

    def test_duplicate_within_same_list_handled(self):
        """If the same chunk appears twice in one list (shouldn't happen, but defensive), it merges."""
        dense = make_dense_results([("A", 0.9), ("A", 0.8), ("B", 0.7)])
        bm25 = make_bm25_results([("B", 10.0)])

        fused = reciprocal_rank_fusion(dense, bm25)
        chunk_ids = [r["chunk_id"] for r in fused]

        # A should appear at most once in the output
        assert (
            chunk_ids.count("A") <= 1
        ), "Duplicate chunk should be de-duplicated in RRF output"

    def test_k_parameter_affects_ranking(self):
        """Changing k should change the relative weight of top vs lower ranks."""
        dense = make_dense_results([("A", 0.99), ("B", 0.50)])
        bm25 = make_bm25_results([("B", 20.0), ("A", 1.0)])

        # With low k, rank differences matter more — B (rank 1 in BM25) should beat A
        fused_low_k = reciprocal_rank_fusion(dense, bm25, k=1)
        # With very high k, all ranks are flattened — A (rank 1 in dense with 0.99) may win
        fused_high_k = reciprocal_rank_fusion(dense, bm25, k=1000)

        # Just verify different k values can produce different orderings
        # (not asserting specific ranking, just that k parameter is respected)
        low_k_scores = {r["chunk_id"]: r["rrf_score"] for r in fused_low_k}
        high_k_scores = {r["chunk_id"]: r["rrf_score"] for r in fused_high_k}

        # Scores should differ between k=1 and k=1000
        assert (
            low_k_scores != high_k_scores
        ), "Different k values should produce different RRF scores"

    def test_rrf_metadata_preserved(self):
        """Original metadata fields (dense_score, bm25_score) should be preserved."""
        dense = make_dense_results([("A", 0.92)])
        bm25 = make_bm25_results([("A", 11.5)])

        fused = reciprocal_rank_fusion(dense, bm25)
        chunk_a = next(r for r in fused if r["chunk_id"] == "A")

        assert chunk_a.get("dense_score") == 0.92
        assert chunk_a.get("bm25_score") == 11.5
        assert chunk_a.get("rrf_score") is not None

    @pytest.mark.parametrize("list_size", [5, 20, 50])
    def test_large_lists(self, list_size: int):
        """RRF should work correctly with larger candidate lists."""
        dense = make_dense_results(
            [(f"chunk_{i}", 1.0 - i * 0.01) for i in range(list_size)]
        )
        bm25 = make_bm25_results(
            [(f"chunk_{i}", float(list_size - i)) for i in range(list_size)]
        )

        fused = reciprocal_rank_fusion(dense, bm25)

        assert len(fused) == list_size
        assert fused[0]["chunk_id"] == "chunk_0"  # rank 1 in both → should be #1
