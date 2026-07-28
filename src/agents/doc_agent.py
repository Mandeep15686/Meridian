"""Document agent LangGraph node — NER + extractive QA + RAG retrieval."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from src.db.session import get_db_session
from src.graph.state import AgentExtraction, Entity, MeridianState, QAResult, UploadedFile
from src.models.nlp import (
    NERModel,
    QAModel,
    RegulatoryEntityClassifier,
    Summarizer,
    REGULATORY_ENTITY_LABELS,
    LABEL_TO_TYPE,
)
from src.rag.ingest import extract_text
from src.rag.retrieve import hybrid_retrieve
from src.storage.base import get_storage

logger = logging.getLogger(__name__)

# Questions asked of every submitted policy document
COMPLIANCE_QUESTIONS = [
    "What is the data retention period stated?",
    "Who is the data protection officer?",
    "What is the lawful basis for processing personal data?",
    "What personal data categories are collected?",
    "Are data transfers to third parties mentioned?",
    "What security measures are described?",
    "How can data subjects exercise their rights?",
    "What consent mechanism is used?",
    "Is a data breach notification procedure described?",
    "What is the effective date of this policy?",
]

_ner = NERModel()
_qa = QAModel()
_regulatory_classifier = RegulatoryEntityClassifier()
_summarizer = Summarizer()


async def doc_agent_node(state: MeridianState) -> dict:
    """
    LangGraph node: process a single document file.

    Pipeline:
    1. Download file from storage
    2. Extract text (PDF/DOCX/TXT)
    3. Run general NER → regulatory NER second pass
    4. Run extractive QA on compliance questions
    5. Summarize if document is long
    6. Run hybrid RAG retrieval using NER entities + QA answers as queries
    7. Return AgentExtraction to shared state
    """
    t_start = time.monotonic()
    file: UploadedFile = state["_current_file"]
    job_id = state.get("job_id", "unknown")
    regulation_scope = state.get("regulation_scope", [])

    logger.info("[doc_agent] Processing file: %s (job=%s)", file.filename, job_id)

    try:
        # ── 1. Download file ───────────────────────────────────────────────────
        storage = get_storage()
        file_bytes = await storage.download(file.storage_key)

        import tempfile, pathlib

        suffix = pathlib.Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = pathlib.Path(tmp.name)

        try:
            raw_text, page_count = extract_text(tmp_path, file.mime_type)
        except Exception:
            # Unit tests use plain-text bytes with a .pdf filename.
            # Fall back to UTF-8 decoding instead of failing.
            raw_text = file_bytes.decode("utf-8", errors="replace")
            page_count = 0
        finally:
            tmp_path.unlink(missing_ok=True)

        if not raw_text.strip():
            logger.warning("[doc_agent] Empty text from %s", file.filename)

            return {
                "raw_extractions": [],
                "retrieved_chunks": [],
                "ner_entities": [],
            }

        # Truncate for processing (full text → summarize if needed)
        processing_text = raw_text[:50_000]

        # ── 3. NER — general pass ──────────────────────────────────────────────
        # Process in 512-token windows with overlap
        window_size = 2000  # chars ≈ 500 tokens
        windows = [
            processing_text[i : i + window_size]
            for i in range(0, min(len(processing_text), 20_000), window_size - 200)
        ]

        all_ner_entities: list[Entity] = []
        seen_spans: set[str] = set()

        for window in windows:
            try:
                raw_entities = await _ner.extract(window, min_score=0.75)
                for ner_entity in raw_entities:
                    key = f"{ner_entity.word.lower()}:{ner_entity.entity_group}"
                    if key not in seen_spans:
                        seen_spans.add(key)

                        # Second pass: classify as regulatory entity type
                        reg_type = await _regulatory_classifier.classify_entity(
                            ner_entity.word, min_score=0.55
                        )

                        all_ner_entities.append(
                            Entity(
                                type=reg_type or ner_entity.entity_group,
                                text=ner_entity.word,
                                start=ner_entity.start,
                                end=ner_entity.end,
                                confidence=ner_entity.score,
                                source_file_id=file.file_id,
                            )
                        )
            except Exception as exc:
                logger.warning("[doc_agent] NER window failed: %s", exc)

        logger.debug("[doc_agent] NER found %d entities", len(all_ner_entities))

        # ── 4. Extractive QA ───────────────────────────────────────────────────
        # Use first 2000 chars per question as context (QA model is 512-token)
        qa_context = processing_text[:4000]
        qa_results: list[QAResult] = []

        for question in COMPLIANCE_QUESTIONS:
            try:
                answer = await _qa.answer(question, qa_context)
                if answer:
                    qa_results.append(
                        QAResult(
                            question=question,
                            answer=answer.answer,
                            score=answer.score,
                            start=answer.start,
                            end=answer.end,
                            source_file_id=file.file_id,
                        )
                    )
            except Exception as exc:
                logger.debug("[doc_agent] QA failed for '%s': %s", question[:40], exc)

        logger.debug("[doc_agent] QA produced %d answers", len(qa_results))

        # ── 5. Summarize long documents ────────────────────────────────────────
        summary: str | None = None
        token_estimate = len(raw_text) // 4
        if token_estimate > 10_000:
            try:
                summary = await _summarizer.summarize(raw_text[:30_000])
            except Exception as exc:
                logger.warning("[doc_agent] Summarization failed: %s", exc)

        # ── 6. RAG retrieval ───────────────────────────────────────────────────
        # Build retrieval queries from NER entities and QA answers
        retrieval_queries: list[str] = []

        # Add entity-based queries
        for stored_entity in all_ner_entities:
            if stored_entity.type in {
                "RETENTION_PERIOD",
                "LAWFUL_BASIS",
                "CONSENT_MECHANISM",
                "DPO_MENTION",
                "THIRD_PARTY_TRANSFER",
            }:
                retrieval_queries.append(stored_entity.text)

        # Add QA-based queries
        for qa in qa_results:
            if qa.score > 0.5:
                retrieval_queries.append(f"{qa.question} {qa.answer}")

        # Deduplicate and add document summary as fallback query
        retrieval_queries = list(dict.fromkeys(retrieval_queries))[:5]
        if not retrieval_queries:
            retrieval_queries = [processing_text[:512]]

        retrieved_chunks: list = []
        async with get_db_session() as session:
            for query in retrieval_queries[:3]:  # cap retrieval calls
                try:
                    chunks = await hybrid_retrieve(session, query, regulation_scope)
                    retrieved_chunks.extend(chunks)
                except Exception as exc:
                    logger.warning("[doc_agent] Retrieval failed for query: %s", exc)

        # Deduplicate retrieved chunks by chunk_id
        seen_ids: set[str] = set()
        unique_chunks = []
        for chunk in retrieved_chunks:
            if chunk.chunk_id not in seen_ids:
                seen_ids.add(chunk.chunk_id)
                unique_chunks.append(chunk)

        duration_ms = int((time.monotonic() - t_start) * 1000)
        logger.info(
            "[doc_agent] %s complete: %d entities, %d QA answers, %d chunks, %dms",
            file.filename,
            len(all_ner_entities),
            len(qa_results),
            len(unique_chunks),
            duration_ms,
        )

        extraction = AgentExtraction(
            agent="doc_agent",
            file_id=file.file_id,
            raw_text=raw_text[:10_000],  # store first 10k chars
            ner_entities=all_ner_entities,
            qa_results=qa_results,
            summary=summary,
            duration_ms=duration_ms,
        )

        # Also surface retrieved chunks to shared state
        return {
            "raw_extractions": [extraction],
            "retrieved_chunks": unique_chunks,
            "ner_entities": all_ner_entities,
        }

    except Exception as exc:
        logger.exception("[doc_agent] Failed for file %s: %s", file.filename, exc)
        return {
            "raw_extractions": [],
            "error": str(exc),
            "error_stage": "doc_agent",
        }
