"""
Retrieval pipeline diagnostic tool.

Tests each retrieval stage independently and shows what chunks are returned
for a given query. Used to debug poor retrieval quality.

Usage:
    python scripts/diagnose_retrieval.py --query "data retention period" --scope gdpr
    python scripts/diagnose_retrieval.py --query "lawful basis" --scope gdpr --top-k 10
    python scripts/diagnose_retrieval.py --job-id 01JTEST000 --query "consent mechanism"
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
app = typer.Typer()


async def _run_diagnostic(
    query: str,
    scope: list[str],
    top_k: int,
    verbose: bool,
) -> None:
    """Execute the full retrieval pipeline and show results at each stage."""
    from src.db.session import get_db_session
    from src.models.retrieval import CrossEncoderReranker
    from src.rag.retrieve import (
        bm25_retrieve,
        dense_retrieve,
        embed_query,
        reciprocal_rank_fusion,
    )

    reranker = CrossEncoderReranker()

    console.print("\n[bold]Retrieval diagnostic[/bold]")
    console.print(f"Query: [cyan]{query}[/cyan]")
    console.print(f"Scope: {scope}")
    console.print(f"Top-k: {top_k}\n")

    async with get_db_session() as db:
        # ── Stage 1: Embed query ──────────────────────────────────────────────
        console.print("[bold]Stage 1:[/bold] Embedding query...")
        from src.config import settings

        query_embedding = await embed_query(query)
        console.print(f"  Embedding dimensions: {len(query_embedding)}")
        console.print(f"  Model: {settings.EMBEDDING_MODEL}")

        # ── Stage 2: Dense retrieval ──────────────────────────────────────────
        console.print(f"\n[bold]Stage 2:[/bold] Dense retrieval (top {top_k * 2})...")
        dense_results = await dense_retrieve(db, query_embedding, scope, top_k * 2)
        console.print(f"  Returned: {len(dense_results)} chunks")

        if verbose and dense_results:
            dense_table = Table(show_header=True, show_lines=True)
            dense_table.add_column("Rank", width=6)
            dense_table.add_column("Chunk ID", width=16)
            dense_table.add_column("Score", width=8)
            dense_table.add_column("Article", width=20)
            dense_table.add_column("Content (first 80 chars)", width=80)
            for r in dense_results[:10]:
                dense_table.add_row(
                    str(r["dense_rank"]),
                    r["chunk_id"][:16],
                    f"{r['dense_score']:.4f}",
                    r.get("article") or "—",
                    r["content"][:80],
                )
            console.print(dense_table)

        # ── Stage 3: BM25 retrieval ───────────────────────────────────────────
        console.print(
            f"\n[bold]Stage 3:[/bold] BM25 keyword retrieval (top {top_k * 2})..."
        )
        bm25_results = await bm25_retrieve(db, query, scope, top_k * 2)
        console.print(f"  Returned: {len(bm25_results)} chunks")

        if verbose and bm25_results:
            bm25_table = Table(show_header=True, show_lines=True)
            bm25_table.add_column("Rank", width=6)
            bm25_table.add_column("Chunk ID", width=16)
            bm25_table.add_column("BM25 Score", width=10)
            bm25_table.add_column("Article", width=20)
            bm25_table.add_column("Content (first 80 chars)", width=80)
            for r in bm25_results[:10]:
                bm25_table.add_row(
                    str(r["bm25_rank"]),
                    r["chunk_id"][:16],
                    f"{r['bm25_score']:.4f}",
                    r.get("article") or "—",
                    r["content"][:80],
                )
            console.print(bm25_table)

        # Overlap analysis
        dense_ids = {r["chunk_id"] for r in dense_results}
        bm25_ids = {r["chunk_id"] for r in bm25_results}
        overlap = dense_ids & bm25_ids
        console.print(f"\n  Overlap between dense and BM25: {len(overlap)} chunks")
        if verbose and overlap:
            console.print(f"  Overlapping chunk IDs: {', '.join(list(overlap)[:5])}")

        # ── Stage 4: RRF fusion ───────────────────────────────────────────────
        console.print("\n[bold]Stage 4:[/bold] Reciprocal Rank Fusion...")
        fused = reciprocal_rank_fusion(dense_results, bm25_results)
        top_20 = fused[:20]
        console.print(
            f"  Merged to: {len(fused)} unique chunks → top {len(top_20)} for reranking"
        )

        # ── Stage 5: Cross-encoder reranking ──────────────────────────────────
        console.print(
            f"\n[bold]Stage 5:[/bold] Cross-encoder reranking (top {top_k})..."
        )
        reranked = await reranker.rerank(query=query, candidates=top_20, top_k=top_k)
        console.print(f"  Final ranked: {len(reranked)} chunks\n")

    # ── Final results ─────────────────────────────────────────────────────────
    console.print(Panel("[bold green]Final retrieved chunks[/bold green]"))
    results_table = Table(show_header=True, show_lines=True)
    results_table.add_column("Final rank", width=10)
    results_table.add_column("Chunk ID", width=20)
    results_table.add_column("Rerank score", width=12)
    results_table.add_column("Article", width=24)
    results_table.add_column("Content preview", width=70)

    for chunk in reranked:
        results_table.add_row(
            str(chunk.original_rank + 1),
            chunk.chunk_id[:20],
            f"{chunk.rerank_score:.4f}",
            next(
                (
                    r.get("article") or "—"
                    for r in fused
                    if r["chunk_id"] == chunk.chunk_id
                ),
                "—",
            ),
            chunk.content[:70].replace("\n", " "),
        )
    console.print(results_table)

    # Quality assessment
    if not reranked:
        console.print(
            "\n[red]⚠ No chunks retrieved — check corpus ingestion status[/red]"
        )
    elif reranked[0].rerank_score < 0.1:
        console.print(
            f"\n[yellow]⚠ Top rerank score is low ({reranked[0].rerank_score:.4f}) "
            f"— query may not match corpus content well[/yellow]"
        )
    else:
        console.print(
            f"\n[green]✓ Top chunk score: {reranked[0].rerank_score:.4f}[/green]"
        )


@app.command()
def main(
    query: str = typer.Option(..., "--query", "-q", help="Retrieval query to test"),
    scope: list[str] = typer.Option(
        ["gdpr"], "--scope", "-s", help="Regulation scope(s) to search"
    ),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of chunks to return"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show per-stage results"
    ),
    job_id: str | None = typer.Option(
        None,
        "--job-id",
        help="If set, also shows which chunks were retrieved for that specific job",
    ),
) -> None:
    """Diagnose the retrieval pipeline for a specific query."""
    asyncio.run(_run_diagnostic(query=query, scope=scope, top_k=top_k, verbose=verbose))


if __name__ == "__main__":
    app()
