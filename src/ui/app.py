"""Meridian Streamlit demo dashboard — file upload, job tracking, report viewer."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import cast

import httpx
import streamlit as st

# ── Configuration ─────────────────────────────────────────────────────────────

API_BASE = os.getenv("MERIDIAN_API_URL", "http://localhost:8000")
API_KEY = os.getenv("MERIDIAN_API_KEY", "")

VALID_SCOPES = ["gdpr", "soc2", "iso27001", "sec_sp", "cfpb"]

st.set_page_config(
    page_title="Meridian — Compliance Intelligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── API client ────────────────────────────────────────────────────────────────


def api_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {API_KEY}"}


def api_get(path: str) -> dict:
    with httpx.Client(base_url=API_BASE, timeout=30.0) as client:
        resp = client.get(path, headers=api_headers())
        resp.raise_for_status()
        return cast(dict, resp.json())


def api_submit(files: list, regulation_scope: list[str], webhook_url: str | None) -> dict:
    with httpx.Client(base_url=API_BASE, timeout=60.0) as client:
        multipart_files = [
            ("files", (f.name, f.read(), f.type or "application/octet-stream")) for f in files
        ]
        form_data: dict[str, str] = {"regulation_scope": ",".join(regulation_scope)}
        if webhook_url:
            form_data["webhook_url"] = webhook_url
        resp = client.post(
            "/v1/submit",
            headers={"Authorization": f"Bearer {API_KEY}"},
            files=multipart_files,
            data=form_data,
        )
        resp.raise_for_status()
        return cast(dict, resp.json())


# ── Page: Submit ──────────────────────────────────────────────────────────────


def page_submit() -> None:
    st.header("Submit compliance review")
    st.markdown(
        "Upload one or more files — PDF policies, audio recordings, "
        "compliance screenshots, or CSV audit logs."
    )

    with st.form("submit_form"):
        uploaded_files = st.file_uploader(
            "Upload files",
            accept_multiple_files=True,
            type=[
                "pdf",
                "docx",
                "txt",
                "mp3",
                "wav",
                "m4a",
                "png",
                "jpg",
                "jpeg",
                "webp",
                "csv",
                "xlsx",
            ],
        )

        regulation_scope = st.multiselect(
            "Regulatory frameworks",
            options=VALID_SCOPES,
            default=["gdpr"],
            help="Select one or more regulatory frameworks to check against.",
        )

        webhook_url = st.text_input(
            "Webhook URL (optional)",
            placeholder="https://yourapp.com/webhooks/meridian",
            help="Receive a POST notification when the job completes.",
        )

        submitted = st.form_submit_button("Run compliance review", type="primary")

    if submitted:
        if not uploaded_files:
            st.error("Please upload at least one file.")
            return
        if not regulation_scope:
            st.error("Please select at least one regulatory framework.")
            return

        with st.spinner("Submitting job..."):
            try:
                result = api_submit(
                    uploaded_files,
                    regulation_scope,
                    webhook_url or None,
                )
                st.success(f"Job submitted! Job ID: **{result['job_id']}**")
                st.session_state["last_job_id"] = result["job_id"]
                st.info(f"Poll status at: `{result['poll_url']}`")
            except httpx.HTTPStatusError as exc:
                st.error(f"Submission failed: {exc.response.text}")
            except Exception as exc:
                st.error(f"Error: {exc}")

    # Demo button
    st.divider()
    st.subheader("Demo scenario")
    st.markdown(
        "Load a sample privacy policy and run it against the GDPR corpus to see Meridian in action."
    )
    if st.button("Run demo (Meta-style privacy policy vs GDPR)"):
        sample_path = Path("data/sample_docs/sample_privacy_policy.pdf")
        if sample_path.exists():
            with open(sample_path, "rb") as f:
                sample_bytes = f.read()

            class _FakeFile:
                name = "sample_privacy_policy.pdf"
                type = "application/pdf"

                def read(self) -> bytes:
                    return sample_bytes

            try:
                result = api_submit([_FakeFile()], ["gdpr"], None)
                st.success(f"Demo job submitted! Job ID: **{result['job_id']}**")
                st.session_state["last_job_id"] = result["job_id"]
            except Exception as exc:
                st.error(f"Demo failed: {exc}")
        else:
            st.warning(
                "Sample policy not found at data/sample_docs/sample_privacy_policy.pdf. "
                "Upload your own file above."
            )


# ── Page: Status ──────────────────────────────────────────────────────────────


def page_status() -> None:
    st.header("Job status")

    # Pre-fill from session state
    default_id = st.session_state.get("last_job_id", "")
    job_id = st.text_input("Job ID", value=default_id, placeholder="01JXXXXXXXXXXXXXXXXXXXXXXX")

    if not job_id:
        st.info("Enter a job ID to check status.")
        return

    col1, col2 = st.columns([1, 4])
    with col1:
        auto_refresh = st.checkbox("Auto-refresh every 5s", value=True)
    with col2:
        if st.button("Refresh now"):
            st.rerun()

    try:
        status = api_get(f"/v1/status/{job_id}")
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            st.error(f"Job `{job_id}` not found or expired.")
        else:
            st.error(f"API error: {exc.response.text}")
        return
    except Exception as exc:
        st.error(f"Connection error: {exc}")
        return

    # Status indicator
    job_status = status["status"]
    status_colors = {
        "queued": "🟡",
        "processing": "🔵",
        "complete": "🟢",
        "failed": "🔴",
        "cancelled": "⚫",
    }
    icon = status_colors.get(job_status, "⚪")
    st.subheader(f"{icon} Status: **{job_status.upper()}**")

    # Metadata
    cols = st.columns(4)
    cols[0].metric(
        "Submitted", status.get("submitted_at", "—")[:19] if status.get("submitted_at") else "—"
    )
    cols[1].metric(
        "Started", status.get("started_at", "—")[:19] if status.get("started_at") else "—"
    )
    cols[2].metric(
        "Completed", status.get("completed_at", "—")[:19] if status.get("completed_at") else "—"
    )
    cols[3].metric(
        "Duration",
        f"{status.get('duration_seconds', 0)}s" if status.get("duration_seconds") else "—",
    )

    if job_status == "processing":
        current_stage = status.get("current_stage", "unknown")
        progress = status.get("progress_pct", 0) or 0
        st.progress(progress / 100, text=f"Running: {current_stage} ({progress}%)")

    if job_status == "complete":
        summary = status.get("summary") or {}
        if summary:
            st.success(f"Found **{summary.get('total_gaps', 0)} compliance gaps**")
            gap_cols = st.columns(3)
            by_sev = summary.get("by_severity", {})
            gap_cols[0].metric("Critical", by_sev.get("critical", 0))
            gap_cols[1].metric("Major", by_sev.get("major", 0))
            gap_cols[2].metric("Minor", by_sev.get("minor", 0))

        if status.get("langsmith_trace_url"):
            st.link_button("View LangSmith trace", status["langsmith_trace_url"])

        if st.button("View full report →", type="primary"):
            st.session_state["view_report_job_id"] = job_id
            st.session_state["page"] = "report"
            st.rerun()

    if job_status == "failed":
        error = status.get("error", {})
        st.error(
            f"**Error:** {error.get('message', 'Unknown error')} "
            f"(stage: {error.get('stage', 'unknown')}, "
            f"retries: {error.get('retry_count', 0)})"
        )

    # Stages progress
    if status.get("stages_complete"):
        with st.expander("Completed stages"):
            for stage in status["stages_complete"]:
                st.markdown(f"✓ `{stage}`")

    # Auto-refresh
    if auto_refresh and job_status in ("queued", "processing"):
        time.sleep(5)
        st.rerun()


# ── Page: Report ──────────────────────────────────────────────────────────────


def page_report() -> None:
    st.header("Compliance report")

    default_id = st.session_state.get("view_report_job_id", st.session_state.get("last_job_id", ""))
    job_id = st.text_input("Job ID", value=default_id, placeholder="01JXXXXXXXXXXXXXXXXXXXXXXX")

    if not job_id:
        st.info("Enter a job ID to view its report.")
        return

    try:
        report = api_get(f"/v1/report/{job_id}?format=json")
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 409:
            st.warning("Job is still processing. Check Status tab.")
        elif exc.response.status_code == 404:
            st.error("Job not found or expired.")
        else:
            st.error(f"API error: {exc.response.text}")
        return

    # Executive summary
    st.subheader("Executive summary")
    st.markdown(report.get("executive_summary", "—"))

    # Gap stats
    gaps = report.get("gaps", [])
    if not gaps:
        st.success("✓ No compliance gaps detected.")
        return

    critical_gaps = [g for g in gaps if g["severity"] == "critical"]
    major_gaps = [g for g in gaps if g["severity"] == "major"]
    minor_gaps = [g for g in gaps if g["severity"] == "minor"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total gaps", len(gaps))
    col2.metric("Critical", len(critical_gaps), delta_color="inverse")
    col3.metric("Major", len(major_gaps))
    col4.metric("Minor", len(minor_gaps))

    # Compliance scores
    if report.get("compliance_score"):
        st.subheader("Compliance scores")
        score_cols = st.columns(len(report["compliance_score"]))
        for i, (fw, score_data) in enumerate(report["compliance_score"].items()):
            score_cols[i].metric(
                fw.upper(),
                f"{score_data.get('score', 0) * 100:.0f}%",
                help=f"{score_data.get('gaps', 0)} gaps in {score_data.get('checks_performed', 0)} checks",
            )

    # Gap table
    st.subheader("Identified gaps")
    severity_icons = {"critical": "🔴", "major": "🟡", "minor": "⚪"}

    for i, gap in enumerate(gaps, 1):
        icon = severity_icons.get(gap["severity"], "⚪")
        uncertain_label = " ⚠️ *uncertain*" if gap.get("is_uncertain") else ""
        with st.expander(
            f"{icon} [{gap['severity'].upper()}] {gap['regulatory_article']}{uncertain_label}"
        ):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Framework**")
                st.code(gap["framework"].upper())
                st.markdown("**Regulatory requirement**")
                st.info(gap["regulatory_requirement"])
                st.markdown("**Regulatory quote**")
                st.caption(f"> {gap['regulatory_quote'][:300]}")

            with col2:
                st.markdown("**Policy reference**")
                st.code(gap.get("policy_reference") or "Section not found")
                st.markdown("**Policy text**")
                st.warning(gap.get("policy_text") or "_Not present in policy_")
                st.markdown("**Gap description**")
                st.error(gap["gap_description"])

            st.markdown("**Remediation**")
            st.success(gap["remediation"])

            metric_cols = st.columns(3)
            metric_cols[0].metric("Confidence", f"{gap.get('confidence', 0):.0%}")
            metric_cols[1].metric("Groundedness", f"{gap.get('groundedness_score', 0):.0%}")
            metric_cols[2].metric("Severity", gap["severity"].upper())

    # Model metadata
    if report.get("model_metadata"):
        with st.expander("Model metadata (audit trail)"):
            st.json(report["model_metadata"])

    # Download buttons
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        report_json = json.dumps(report, indent=2, default=str)
        st.download_button(
            "Download JSON report",
            data=report_json,
            file_name=f"meridian_report_{job_id}.json",
            mime="application/json",
        )


# ── Sidebar ───────────────────────────────────────────────────────────────────


def render_sidebar() -> str:
    with st.sidebar:
        st.image("https://via.placeholder.com/200x60?text=Meridian", width=200)
        st.caption(f"API: `{API_BASE}`")

        page = cast(
            str,
            st.radio(
            "Navigation",
            options=["Submit", "Status", "Report"],
            index=0,
            ),
        )

        st.divider()
        st.markdown("**Quick links**")
        st.markdown(f"- [API docs]({API_BASE}/docs)")
        st.markdown("- [GitHub](https://github.com/yourhandle/meridian)")

        if st.session_state.get("last_job_id"):
            st.divider()
            st.caption(f"Last job: `{st.session_state['last_job_id'][:12]}...`")

    return page


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    # Check API key
    if not API_KEY:
        st.error(
            "MERIDIAN_API_KEY environment variable is not set. "
            "Set it to your Meridian API key to use the dashboard."
        )
        st.code("export MERIDIAN_API_KEY=mer_live_yourkey")
        return

    page = render_sidebar()

    # Allow override from session state (e.g. "View report" button)
    if st.session_state.get("page"):
        page = st.session_state.pop("page")

    if page == "Submit":
        page_submit()
    elif page == "Status":
        page_status()
    elif page == "Report":
        page_report()


if __name__ == "__main__":
    main()
