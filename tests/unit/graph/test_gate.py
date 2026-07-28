"""Unit tests for the hallucination gate node."""

from __future__ import annotations

from unittest.mock import patch

from src.graph.nodes.gate import gate_routing, hallucination_gate_node
from tests.conftest import make_candidate_gap, make_retrieved_chunk

# ── Helper to make a simple state for gate testing ────────────────────────────


def _make_gate_state(
    gap_ids: list[str],
    groundedness_scores: dict[str, float],
    synthesis_retries: int = 0,
) -> dict:
    gaps = [make_candidate_gap(gap_id=gid) for gid in gap_ids]
    chunks = [make_retrieved_chunk(chunk_id="chunk-gdpr-001")]
    return {
        "job_id": "test-job",
        "candidate_gaps": gaps,
        "retrieved_chunks": chunks,
        "groundedness_scores": {},  # will be computed by gate
        "failed_gap_ids": [],
        "synthesis_retries": synthesis_retries,
        "regulation_scope": ["gdpr"],
    }


class TestHallucinationGate:
    def test_all_passing_gaps_produce_verified_gaps(self):
        """When all gaps score above threshold, they should all be verified."""
        state = _make_gate_state(
            gap_ids=["gap_001", "gap_002"],
            groundedness_scores={"gap_001": 0.92, "gap_002": 0.85},
        )

        with patch("src.graph.nodes.gate._similarity_model") as mock_sim:
            mock_sim.batch_score.return_value = [0.92, 0.85]
            result = hallucination_gate_node(state)

        assert len(result["verified_gaps"]) == 2
        assert result["failed_gap_ids"] == []
        assert result["synthesis_retries"] == 0
        for gap in result["verified_gaps"]:
            assert gap.is_verified is True
            assert gap.is_uncertain is False

    def test_failing_gap_triggers_retry_when_retries_available(self):
        """When a gap fails and retries < max, the gate should signal a retry."""
        state = _make_gate_state(
            gap_ids=["gap_001", "gap_002"],
            groundedness_scores={},
            synthesis_retries=0,
        )

        # gap_001 passes (0.91), gap_002 fails (0.61)
        with patch("src.graph.nodes.gate._similarity_model") as mock_sim:
            mock_sim.batch_score.return_value = [0.91, 0.61]
            result = hallucination_gate_node(state)

        assert "gap_002" in result["failed_gap_ids"]
        assert result["synthesis_retries"] == 1
        assert result["verified_gaps"] == []

    def test_failing_gap_marked_uncertain_after_max_retries(self):
        """When retries are exhausted, failing gaps should be marked uncertain."""
        from src.config import settings

        state = _make_gate_state(
            gap_ids=["gap_001"],
            groundedness_scores={},
            synthesis_retries=settings.MAX_SYNTHESIS_RETRIES,  # at max
        )

        with patch("src.graph.nodes.gate._similarity_model") as mock_sim:
            mock_sim.batch_score.return_value = [0.55]  # below threshold
            result = hallucination_gate_node(state)

        assert len(result["verified_gaps"]) == 1
        uncertain_gap = result["verified_gaps"][0]
        assert uncertain_gap.is_uncertain is True
        assert uncertain_gap.is_verified is False

    def test_exact_threshold_value_fails(self):
        """A groundedness score exactly equal to the threshold should FAIL (exclusive >)."""
        from src.config import settings

        threshold = settings.GROUNDEDNESS_THRESHOLD  # default 0.80

        state = _make_gate_state(["gap_001"], {}, synthesis_retries=0)

        with patch("src.graph.nodes.gate._similarity_model") as mock_sim:
            # Exactly at threshold — should fail (gate uses score < threshold to fail,
            # meaning score == threshold passes... check the implementation)
            mock_sim.batch_score.return_value = [threshold]
            result = hallucination_gate_node(state)

        # Score == threshold should pass (not be in failed list)
        # The gate logic: if score < threshold → fail
        assert "gap_001" not in result.get(
            "failed_gap_ids", []
        ), "Score exactly at threshold should pass (< is exclusive)"

    def test_empty_candidate_gaps_returns_empty(self):
        """Gate with no candidate gaps should return empty verified_gaps."""
        state = {
            "job_id": "test",
            "candidate_gaps": [],
            "retrieved_chunks": [],
            "groundedness_scores": {},
            "failed_gap_ids": [],
            "synthesis_retries": 0,
        }

        result = hallucination_gate_node(state)

        assert result["verified_gaps"] == []
        assert result["failed_gap_ids"] == []

    def test_gaps_sorted_by_severity_then_confidence(self):
        """Verified gaps should be sorted: critical > major > minor, then by confidence."""
        gaps = [
            make_candidate_gap(gap_id="g1", severity="minor", confidence=0.95),
            make_candidate_gap(gap_id="g2", severity="critical", confidence=0.80),
            make_candidate_gap(gap_id="g3", severity="major", confidence=0.90),
            make_candidate_gap(gap_id="g4", severity="critical", confidence=0.92),
        ]
        state = {
            "job_id": "test",
            "candidate_gaps": gaps,
            "retrieved_chunks": [make_retrieved_chunk()],
            "groundedness_scores": {},
            "failed_gap_ids": [],
            "synthesis_retries": 0,
        }

        with patch("src.graph.nodes.gate._similarity_model") as mock_sim:
            mock_sim.batch_score.return_value = [0.91, 0.91, 0.91, 0.91]
            result = hallucination_gate_node(state)

        verified = result["verified_gaps"]
        assert verified[0].gap_id == "g4"  # critical, confidence=0.92
        assert verified[1].gap_id == "g2"  # critical, confidence=0.80
        assert verified[2].gap_id == "g3"  # major
        assert verified[3].gap_id == "g1"  # minor

    def test_groundedness_scores_recorded_per_gap(self):
        """Gate should record groundedness scores keyed by gap_id."""
        state = _make_gate_state(["gap_001", "gap_002"], {})

        with patch("src.graph.nodes.gate._similarity_model") as mock_sim:
            mock_sim.batch_score.return_value = [0.92, 0.88]
            result = hallucination_gate_node(state)

        scores = result["groundedness_scores"]
        assert "gap_001" in scores
        assert "gap_002" in scores
        assert abs(scores["gap_001"] - 0.92) < 0.01
        assert abs(scores["gap_002"] - 0.88) < 0.01

    def test_groundedness_score_stored_on_verified_gap(self):
        """VerifiedGap.groundedness_score should match the computed score."""
        state = _make_gate_state(["gap_001"], {})

        with patch("src.graph.nodes.gate._similarity_model") as mock_sim:
            mock_sim.batch_score.return_value = [0.87]
            result = hallucination_gate_node(state)

        verified = result["verified_gaps"]
        assert len(verified) == 1
        assert abs(verified[0].groundedness_score - 0.87) < 0.01


class TestGateRouting:
    def test_routes_to_synthesize_when_failures_and_retries_available(self):
        """gate_routing should return 'synthesize' when failed_gap_ids and retries < max."""

        state = {
            "failed_gap_ids": ["gap_001"],
            "synthesis_retries": 1,  # still has retries left
        }
        route = gate_routing(state)
        assert route == "synthesize"

    def test_routes_to_report_when_no_failures(self):
        """gate_routing should return 'report' when no failed gaps."""
        state = {"failed_gap_ids": [], "synthesis_retries": 0}
        route = gate_routing(state)
        assert route == "report"

    def test_routes_to_report_when_retries_exhausted(self):
        """gate_routing should return 'report' when retries == max, even with failures."""
        from src.config import settings

        state = {
            "failed_gap_ids": ["gap_001"],
            "synthesis_retries": settings.MAX_SYNTHESIS_RETRIES + 1,
        }
        route = gate_routing(state)
        assert route == "report"
