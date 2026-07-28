"""Unit tests for API Pydantic schemas."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.api.schemas.all import (
    GapResponse,
    JobStatusResponse,
    SubmitOptions,
    SubmitRequest,
)


class TestSubmitRequest:
    def test_valid_single_scope(self):
        req = SubmitRequest(regulation_scope=["gdpr"])
        assert req.regulation_scope == ["gdpr"]

    def test_valid_multiple_scopes(self):
        req = SubmitRequest(regulation_scope=["gdpr", "soc2", "sec_sp"])
        assert len(req.regulation_scope) == 3

    def test_invalid_scope_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            SubmitRequest(regulation_scope=["invalid_framework"])
        assert "Unknown regulation scope" in str(exc_info.value)

    def test_webhook_url_must_be_https(self):
        with pytest.raises(ValidationError) as exc_info:
            SubmitRequest(
                regulation_scope=["gdpr"],
                webhook_url="http://insecure.example.com/hook",
            )
        assert "HTTPS" in str(exc_info.value)

    def test_valid_https_webhook(self):
        req = SubmitRequest(
            regulation_scope=["gdpr"],
            webhook_url="https://secure.example.com/hook",
        )
        assert req.webhook_url == "https://secure.example.com/hook"

    def test_none_webhook_url_allowed(self):
        req = SubmitRequest(regulation_scope=["gdpr"], webhook_url=None)
        assert req.webhook_url is None

    def test_invalid_report_format_raises(self):
        with pytest.raises(ValidationError):
            SubmitRequest(
                regulation_scope=["gdpr"],
                report_formats=["xml"],  # not a valid format
            )

    def test_default_report_formats(self):
        req = SubmitRequest(regulation_scope=["gdpr"])
        assert "pdf" in req.report_formats
        assert "json" in req.report_formats

    def test_default_options(self):
        req = SubmitRequest(regulation_scope=["gdpr"])
        assert req.options.groundedness_threshold == 0.80
        assert req.options.max_synthesis_retries == 2
        assert req.options.include_regulatory_appendix is True

    def test_custom_options(self):
        req = SubmitRequest(
            regulation_scope=["gdpr"],
            options=SubmitOptions(
                groundedness_threshold=0.90,
                max_synthesis_retries=3,
                severity_filter=["critical"],
            ),
        )
        assert req.options.groundedness_threshold == 0.90
        assert req.options.severity_filter == ["critical"]

    def test_groundedness_threshold_bounds(self):
        with pytest.raises(ValidationError):
            SubmitOptions(groundedness_threshold=1.5)  # > 1.0
        with pytest.raises(ValidationError):
            SubmitOptions(groundedness_threshold=-0.1)  # < 0.0

    def test_language_must_be_two_letters(self):
        req = SubmitRequest(regulation_scope=["gdpr"], language="fr")
        assert req.language == "fr"

        with pytest.raises(ValidationError):
            SubmitRequest(regulation_scope=["gdpr"], language="fra")  # 3 letters

    def test_all_valid_scopes_accepted(self):
        valid_scopes = ["gdpr", "soc2", "iso27001", "sec_sp", "cfpb", "sec_sid", "eu_ai_act"]
        req = SubmitRequest(regulation_scope=valid_scopes)
        assert set(req.regulation_scope) == set(valid_scopes)


class TestGapResponse:
    def test_valid_critical_gap(self):
        gap = GapResponse(
            gap_id="gap_001",
            severity="critical",
            framework="gdpr",
            regulatory_article="GDPR Article 13(2)(a)",
            regulatory_requirement="Specify data retention period.",
            regulatory_quote="The period for which personal data will be stored...",
            gap_description="No retention period specified.",
            severity_justification="Frequently cited GDPR violation.",
            remediation="Add a retention table.",
            confidence=0.91,
            groundedness_score=0.88,
        )
        assert gap.severity == "critical"
        assert gap.is_verified is True
        assert gap.is_uncertain is False

    def test_invalid_severity_raises(self):
        with pytest.raises(ValidationError):
            GapResponse(
                gap_id="gap_001",
                severity="extreme",  # not valid
                framework="gdpr",
                regulatory_article="GDPR Article 13",
                regulatory_requirement="Test",
                regulatory_quote="Test quote",
                gap_description="Test",
                severity_justification="Test",
                remediation="Test",
                confidence=0.9,
                groundedness_score=0.9,
            )

    def test_uncertain_gap(self):
        gap = GapResponse(
            gap_id="gap_002",
            severity="major",
            framework="gdpr",
            regulatory_article="GDPR Article 6",
            regulatory_requirement="Specify lawful basis.",
            regulatory_quote="Processing shall be lawful only if...",
            gap_description="No lawful basis stated.",
            severity_justification="Major gap.",
            remediation="State lawful basis.",
            confidence=0.75,
            groundedness_score=0.72,
            is_verified=False,
            is_uncertain=True,
        )
        assert gap.is_uncertain is True
        assert gap.is_verified is False


class TestJobStatusResponse:
    def test_queued_status(self):
        status = JobStatusResponse(
            job_id="01JTEST",
            status="queued",
            submitted_at=datetime.now(UTC),
        )
        assert status.status == "queued"
        assert status.completed_at is None
        assert status.summary is None

    def test_invalid_status_raises(self):
        with pytest.raises(ValidationError):
            JobStatusResponse(
                job_id="01JTEST",
                status="running",  # not a valid status
                submitted_at=datetime.now(UTC),
            )

    def test_all_valid_statuses(self):
        for valid_status in ["queued", "processing", "complete", "failed", "cancelled"]:
            status = JobStatusResponse(
                job_id="01JTEST",
                status=valid_status,
                submitted_at=datetime.now(UTC),
            )
            assert status.status == valid_status
