"""
LangGraph pipeline definition for Meridian.

Graph topology:
  classify_input
      ↓ (conditional fan-out via Send)
  [doc_agent | audio_agent | vision_agent | data_agent]  (parallel)
      ↓ (all converge to synthesize)
  synthesize
      ↓
  gate  ──(fail + retries left)──→ synthesize (loop)
      ↓ (all pass or retries exhausted)
  report
      ↓
  END
"""

from __future__ import annotations

import logging

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from src.agents.audio_agent import audio_agent_node
from src.agents.data_agent import data_agent_node
from src.agents.doc_agent import doc_agent_node
from src.agents.synthesis_agent import report_node, synthesis_node
from src.agents.vision_agent import vision_agent_node
from src.config import settings
from src.graph.nodes.gate import gate_routing, hallucination_gate_node
from src.graph.router import classify_input_node, route_to_agents
from src.graph.state import MeridianState

logger = logging.getLogger(__name__)


def build_graph(checkpointer_path: str = "checkpoints.db") -> StateGraph:
    """
    Construct and compile the Meridian LangGraph state machine.

    Args:
        checkpointer_path: SQLite path for resumable execution checkpoints.
                           In production, replace with AsyncPostgresSaver.

    Returns:
        A compiled LangGraph graph ready for invocation.
    """
    builder = StateGraph(MeridianState)

    # ── Nodes ──────────────────────────────────────────────────────────────────
    builder.add_node("classify_input", classify_input_node)
    builder.add_node("doc_agent",      doc_agent_node)
    builder.add_node("audio_agent",    audio_agent_node)
    builder.add_node("vision_agent",   vision_agent_node)
    builder.add_node("data_agent",     data_agent_node)
    builder.add_node("synthesize",     synthesis_node)
    builder.add_node("gate",           hallucination_gate_node)
    builder.add_node("report",         report_node)

    # ── Entry point ────────────────────────────────────────────────────────────
    builder.set_entry_point("classify_input")

    # ── Fan-out: classify → specialist agents (parallel via Send) ──────────────
    builder.add_conditional_edges(
        "classify_input",
        route_to_agents,
        # Send targets — explicitly list all possible target nodes
        {
            "doc_agent":    "doc_agent",
            "audio_agent":  "audio_agent",
            "vision_agent": "vision_agent",
            "data_agent":   "data_agent",
            "synthesize":   "synthesize",  # fallback when no agents match
        },
    )

    # ── Fan-in: all specialist agents converge to synthesize ───────────────────
    for agent_node in ("doc_agent", "audio_agent", "vision_agent", "data_agent"):
        builder.add_edge(agent_node, "synthesize")

    # ── Synthesis → gate → (retry loop or report) ──────────────────────────────
    builder.add_edge("synthesize", "gate")
    builder.add_conditional_edges(
        "gate",
        gate_routing,
        {
            "synthesize": "synthesize",
            "report":     "report",
        },
    )

    # ── Report → END ───────────────────────────────────────────────────────────
    builder.add_edge("report", END)

    # ── Checkpointer for resumable execution ───────────────────────────────────
    checkpointer = SqliteSaver.from_conn_string(checkpointer_path)
    graph = builder.compile(checkpointer=checkpointer)

    logger.info("Meridian LangGraph compiled successfully")
    return graph


# Module-level singleton (created once per worker process)
_graph: StateGraph | None = None


def get_graph() -> StateGraph:
    """Return the module-level compiled graph, building it on first call."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


async def run_pipeline(
    job_id: str,
    input_files: list,
    regulation_scope: list[str],
    options: dict | None = None,
) -> MeridianState:
    """
    Execute the Meridian pipeline for a job.

    Args:
        job_id: The ULID job identifier.
        input_files: List of UploadedFile dataclass instances.
        regulation_scope: List of corpus IDs, e.g. ['gdpr', 'soc2'].
        options: Optional pipeline configuration overrides.

    Returns:
        The final MeridianState after the pipeline completes.
    """
    initial_state: MeridianState = {
        "job_id": job_id,
        "input_files": input_files,
        "regulation_scope": regulation_scope,
        "options": options or {},
        "raw_extractions": [],
        "synthesis_retries": 0,
        "metadata": {"pipeline_version": settings.VERSION},
    }

    config = {
        "configurable": {"thread_id": job_id},
        "metadata": {
            "job_id": job_id,
            "regulation_scope": ",".join(regulation_scope),
        },
        "tags": ["meridian", f"job:{job_id}"],
    }

    if settings.LANGCHAIN_TRACING_V2:
        config["metadata"]["langsmith_project"] = settings.LANGCHAIN_PROJECT

    graph = get_graph()
    final_state: MeridianState = await graph.ainvoke(initial_state, config=config)

    if final_state.get("error"):
        logger.error(
            "Pipeline failed for job %s at stage %s: %s",
            job_id,
            final_state.get("error_stage"),
            final_state.get("error"),
        )

    return final_state
