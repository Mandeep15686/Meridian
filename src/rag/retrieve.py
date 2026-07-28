"""
Hybrid retrieval pipeline: dense → BM25 → RRF → cross-encoder rerank.

Three-stage approach:
  Stage 1: Dense vector search via pgvector (cosine similarity)
  Stage 2: BM25 keyword search via PostgreSQL ts_vector + ts_rank_cd
  Stage 3: Reciprocal Rank Fusion → cross-encoder reranking (top-20 → top-5)
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from openai import AsyncOpenAI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.graph.state import RetrievedChunk
from src.models.retrieval import CrossEncoderReranker

logger = logging.getLogger(__name__)

_openai_client = AsyncOpenAI(
    api_key=settings.ANTHROPIC_API_KEY
)  # uses same key pattern
_reranker = CrossEncoderReranker()


# ── Query embedding ───────────────────────────────────────────────────────────


async def embed_query(query: str) -> list[float]:
    """Embed a query string using the configured embedding model."""
    client = (
        AsyncOpenAI()
    )  # reads OPENAI_API_KEY; set OPENAI_API_KEY = ANTHROPIC_API_KEY alias
    response = await client.embeddings.create(
        model=settings.EMBEDDING_MODEL,
        input=query[:8191],
    )
    return response.data[0].embedding


# ── Stage 1: Dense retrieval ──────────────────────────────────────────────────


async def dense_retrieve(
    session: AsyncSession,
    query_embedding: list[float],
    regulation_scope: list[str],
    top_k: int,
) -> list[dict[str, Any]]:
    """
    Retrieve top-k chunks by cosine similarity using pgvector IVFFlat index.
    """
    # Build the embedding literal for pgvector
    vec_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

    scope_filter = ""
    if regulation_scope:
        placeholders = ", ".join(f"'{s}'" for s in regulation_scope)
        scope_filter = f"AND regulation IN ({placeholders})"

    sql = text(
        f"""
        SET ivfflat.probes = :probes;
        SELECT
            id              AS chunk_id,
            regulation,
            article,
            content,
            jurisdiction,
            (1 - (embedding <=> :embedding::vector)) AS dense_score
        FROM chunks
        WHERE embedding IS NOT NULL
        {scope_filter}
        ORDER BY embedding <=> :embedding::vector
        LIMIT :top_k;
    """
    )

    result = await session.execute(
        sql,
        {
            "embedding": vec_str,
            "probes": settings.IVFFLAT_PROBES,
            "top_k": top_k,
        },
    )
    rows = result.mappings().all()

    return [
        {
            "chunk_id": row["chunk_id"],
            "regulation": row["regulation"],
            "article": row["article"],
            "content": row["content"],
            "jurisdiction": row["jurisdiction"],
            "dense_score": float(row["dense_score"]),
            "dense_rank": i + 1,
        }
        for i, row in enumerate(rows)
    ]


# ── Stage 2: BM25 keyword retrieval ──────────────────────────────────────────


async def bm25_retrieve(
    session: AsyncSession,
    query: str,
    regulation_scope: list[str],
    top_k: int,
) -> list[dict[str, Any]]:
    """
    Retrieve top-k chunks using PostgreSQL full-text search (BM25 approximation).
    """
    scope_filter = ""
    if regulation_scope:
        placeholders = ", ".join(f"'{s}'" for s in regulation_scope)
        scope_filter = f"AND regulation IN ({placeholders})"

    sql = text(
        f"""
        SELECT
            id              AS chunk_id,
            regulation,
            article,
            content,
            jurisdiction,
            ts_rank_cd(ts_vector, plainto_tsquery('english', :query)) AS bm25_score
        FROM chunks
        WHERE ts_vector @@ plainto_tsquery('english', :query)
        {scope_filter}
        ORDER BY bm25_score DESC
        LIMIT :top_k;
    """
    )

    result = await session.execute(sql, {"query": query[:512], "top_k": top_k})
    rows = result.mappings().all()

    return [
        {
            "chunk_id": row["chunk_id"],
            "regulation": row["regulation"],
            "article": row["article"],
            "content": row["content"],
            "jurisdiction": row["jurisdiction"],
            "bm25_score": float(row["bm25_score"]),
            "bm25_rank": i + 1,
        }
        for i, row in enumerate(rows)
    ]


# ── Stage 3: Reciprocal Rank Fusion ──────────────────────────────────────────


def reciprocal_rank_fusion(
    dense_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.

    RRF formula: score(d) = Σ 1 / (k + rank(d))
    k=60 is the standard constant that dampens the impact of high ranks.
    """
    # Build index of all unique chunk IDs with their metadata
    chunks_by_id: dict[str, dict[str, Any]] = {}

    for i, chunk in enumerate(dense_results):
        cid = chunk["chunk_id"]
        if cid not in chunks_by_id:
            chunks_by_id[cid] = {
                **chunk,
                "dense_rank": None,
                "bm25_rank": None,
                "rrf_score": 0.0,
            }
        chunks_by_id[cid]["dense_rank"] = i + 1
        chunks_by_id[cid]["dense_score"] = chunk.get("dense_score", 0.0)
        chunks_by_id[cid]["rrf_score"] += 1.0 / (k + i + 1)

    for i, chunk in enumerate(bm25_results):
        cid = chunk["chunk_id"]
        if cid not in chunks_by_id:
            chunks_by_id[cid] = {
                **chunk,
                "dense_rank": None,
                "bm25_rank": None,
                "rrf_score": 0.0,
            }
        chunks_by_id[cid]["bm25_rank"] = i + 1
        chunks_by_id[cid]["bm25_score"] = chunk.get("bm25_score", 0.0)
        chunks_by_id[cid]["rrf_score"] += 1.0 / (k + i + 1)

    # Sort by RRF score descending
    merged = sorted(chunks_by_id.values(), key=lambda c: -c["rrf_score"])
    return merged


# ── Full retrieval pipeline ───────────────────────────────────────────────────


async def hybrid_retrieve(
    session: AsyncSession,
    query: str,
    regulation_scope: list[str],
) -> list[RetrievedChunk]:
    """
    Full three-stage hybrid retrieval pipeline.

    Args:
        session: Async SQLAlchemy session.
        query: Natural language retrieval query.
        regulation_scope: Active regulatory frameworks (used as filter).

    Returns:
        Top-k RetrievedChunk objects ranked by reranker score.
    """
    # Check cache first
    cache_key = _make_cache_key(query, regulation_scope)
    cached = await _get_cached_results(cache_key)
    if cached is not None:
        logger.debug("Retrieval cache hit for query: %s...", query[:40])
        return cached

    # Stage 1: Dense retrieval
    query_embedding = await embed_query(query)
    dense_results = await dense_retrieve(
        session, query_embedding, regulation_scope, settings.RETRIEVAL_TOP_K_DENSE
    )
    logger.debug("Dense retrieval returned %d candidates", len(dense_results))

    # Stage 2: BM25 retrieval
    bm25_results = await bm25_retrieve(
        session, query, regulation_scope, settings.RETRIEVAL_TOP_K_BM25
    )
    logger.debug("BM25 retrieval returned %d candidates", len(bm25_results))

    # Stage 3a: RRF fusion
    fused = reciprocal_rank_fusion(dense_results, bm25_results)
    top_20 = fused[: max(settings.RETRIEVAL_TOP_K_DENSE, settings.RETRIEVAL_TOP_K_BM25)]
    logger.debug("After RRF: %d merged candidates", len(top_20))

    # Stage 3b: Cross-encoder reranking
    reranked = await _reranker.rerank(
        query=query,
        candidates=top_20,
        top_k=settings.RETRIEVAL_TOP_K_RERANK,
    )
    logger.debug("After reranking: %d chunks selected", len(reranked))

    # Build typed output
    results: list[RetrievedChunk] = []
    for rank, rc in enumerate(reranked):
        # Look up original chunk metadata from fused results
        original = next((c for c in top_20 if c["chunk_id"] == rc.chunk_id), {})
        results.append(
            RetrievedChunk(
                chunk_id=rc.chunk_id,
                regulation=original.get("regulation", "unknown"),
                article=original.get("article"),
                content=rc.content,
                jurisdiction=original.get("jurisdiction", "unknown"),
                dense_score=original.get("dense_score", 0.0),
                bm25_score=original.get("bm25_score", 0.0),
                rrf_score=original.get("rrf_score", 0.0),
                rerank_score=rc.rerank_score,
                final_rank=rank + 1,
            )
        )

    await _set_cached_results(cache_key, results)
    return results


def _make_cache_key(query: str, scope: list[str]) -> str:
    payload = json.dumps({"q": query, "scope": sorted(scope)})
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return f"retrieval:{digest}"


async def _get_cached_results(key: str) -> list[RetrievedChunk] | None:
    if settings.RETRIEVAL_CACHE_TTL <= 0:
        return None
    try:
        import redis.asyncio as aioredis

        r = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        raw = await r.get(key)
        if raw:
            data = json.loads(raw)
            return [RetrievedChunk(**item) for item in data]
    except Exception:
        pass
    return None


async def _set_cached_results(key: str, results: list[RetrievedChunk]) -> None:
    if settings.RETRIEVAL_CACHE_TTL <= 0:
        return
    try:
        import redis.asyncio as aioredis

        r = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        serialised = json.dumps([vars(r_) for r_ in results])
        await r.setex(key, settings.RETRIEVAL_CACHE_TTL, serialised)
    except Exception:
        pass
