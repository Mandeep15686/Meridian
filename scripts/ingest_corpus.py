"""
Corpus ingestion script — downloads, chunks, embeds, and upserts regulatory
documents into the Meridian vector store.

Usage:
    python scripts/ingest_corpus.py --source gdpr
    python scripts/ingest_corpus.py --source gdpr,soc2,sec_sp
    python scripts/ingest_corpus.py --source all
    python scripts/ingest_corpus.py --source gdpr --force-reingest
    python scripts/ingest_corpus.py --source gdpr --dev-mode    # first 500 chunks only
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import UTC

import typer
from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()
app = typer.Typer(help="Meridian regulatory corpus ingestion tool")


async def _ensure_corpus_record(session, loader, force_refresh: bool = False) -> str:
    """
    Ensure a corpus record exists in the database.
    Returns the corpus ID.
    """
    import uuid
    from datetime import datetime

    from sqlalchemy import select

    from src.db.models import Corpus

    info = loader.corpus_info()
    result = await session.execute(select(Corpus).where(Corpus.slug == info["slug"]))
    corpus = result.scalar_one_or_none()

    if corpus is None:
        corpus = Corpus(
            id=str(uuid.uuid4()),
            slug=info["slug"],
            name=info["name"],
            jurisdiction=info["jurisdiction"],
            version=info["version"],
            source_url=info.get("source_url"),
            is_active=True,
        )
        session.add(corpus)
        await session.flush()
        console.print(f"  Created corpus record: [cyan]{info['slug']}[/cyan]")
    elif force_refresh:
        corpus.last_refreshed = datetime.now(UTC)

    return corpus.id


async def ingest_corpus(
    corpus_id_str: str,
    force_reingest: bool = False,
    dev_mode: bool = False,
    max_documents: int | None = None,
) -> dict[str, int]:
    """
    Ingest a single corpus by its slug.

    Returns:
        Dict with keys: documents_processed, chunks_inserted, chunks_total, skipped.
    """
    from src.db.session import get_db_session
    from src.rag.corpus.registry import get_loader
    from src.rag.ingest import ingest_document

    loader = get_loader(corpus_id_str)
    console.print(f"\n[bold]Ingesting corpus:[/bold] {loader.CORPUS_NAME}")

    t_start = time.monotonic()
    stats = {
        "documents_processed": 0,
        "chunks_inserted": 0,
        "chunks_total": 0,
        "skipped": 0,
    }

    try:
        documents = loader.load_documents()
    except Exception as exc:
        console.print(f"  [red]Failed to load documents: {exc}[/red]")
        return stats

    if max_documents:
        documents = documents[:max_documents]

    if dev_mode:
        documents = documents[:3]  # minimal set for dev
        console.print(
            f"  [yellow]Dev mode: ingesting {len(documents)} documents only[/yellow]"
        )

    console.print(f"  Found {len(documents)} documents to process")

    import tempfile

    async with get_db_session() as session:
        corpus_db_id = await _ensure_corpus_record(session, loader, force_reingest)
        await session.commit()

    for i, doc in enumerate(documents):
        console.print(f"  [{i + 1}/{len(documents)}] {doc.filename}...")

        try:
            # Write content to a temp file for the ingestion pipeline
            suffix = Path(doc.filename).suffix or ".txt"
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=suffix, delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(doc.content)
                tmp_path = Path(tmp.name)

            try:
                async with get_db_session() as session:
                    result = await ingest_document(
                        session=session,
                        corpus_id=corpus_db_id,
                        regulation=corpus_id_str,
                        jurisdiction=loader.JURISDICTION,
                        source_path=tmp_path,
                        source_url=doc.source_url,
                        article_metadata_fn=loader.get_chunk_metadata,
                        force_reingest=force_reingest,
                    )
            finally:
                tmp_path.unlink(missing_ok=True)

            stats["documents_processed"] += 1
            stats["chunks_inserted"] += result.get("chunks_inserted", 0)
            stats["chunks_total"] += result.get("chunks_total", 0)
            if result.get("skipped"):
                stats["skipped"] += 1

        except Exception as exc:
            console.print(f"    [red]Failed: {exc}[/red]")
            logger.exception("Ingestion failed for %s", doc.filename)

    # Update corpus chunk count
    async with get_db_session() as session:
        from sqlalchemy import func, select, update

        from src.db.models import Chunk, Corpus

        count_result = await session.execute(
            select(func.count(Chunk.id)).where(Chunk.corpus_id == corpus_db_id)
        )
        chunk_count = count_result.scalar() or 0

        await session.execute(
            update(Corpus)
            .where(Corpus.id == corpus_db_id)
            .values(
                chunk_count=chunk_count, document_count=stats["documents_processed"]
            )
        )
        await session.commit()

    duration = time.monotonic() - t_start
    console.print(
        f"  [green]✓[/green] {corpus_id_str}: "
        f"{stats['chunks_inserted']} chunks inserted "
        f"({stats['skipped']} docs skipped) "
        f"in {duration:.1f}s"
    )
    return stats


@app.command()
def main(
    source: str = typer.Option(
        "gdpr",
        "--source",
        "-s",
        help="Comma-separated corpus slugs, or 'all' for everything. "
        "Available: gdpr, soc2, iso27001, sec_sp, sec_sid, cfpb, eu_ai_act",
    ),
    force_reingest: bool = typer.Option(
        False,
        "--force-reingest",
        help="Re-ingest all documents even if content hash already exists.",
    ),
    dev_mode: bool = typer.Option(
        False,
        "--dev-mode",
        help="Ingest only 3 documents per corpus (fast, for local development).",
    ),
    max_documents: int | None = typer.Option(
        None,
        "--max-documents",
        help="Maximum documents per corpus (useful for testing).",
    ),
) -> None:
    """Ingest one or more regulatory corpora into Meridian's vector store."""
    from src.rag.corpus.registry import list_corpora

    available = list_corpora()

    if source.lower() == "all":
        selected = available
    else:
        selected = [s.strip() for s in source.split(",") if s.strip()]

    # Validate
    invalid = set(selected) - set(available)
    if invalid:
        console.print(f"[red]Unknown corpus IDs: {invalid}[/red]")
        console.print(f"Available: {available}")
        raise typer.Exit(1)

    console.print("[bold]Meridian corpus ingestion[/bold]")
    console.print(f"Corpora: {selected}")
    if dev_mode:
        console.print("[yellow]Dev mode enabled — minimal ingestion[/yellow]")
    if force_reingest:
        console.print(
            "[yellow]Force re-ingest enabled — all documents will be re-processed[/yellow]"
        )

    total_stats = {"documents_processed": 0, "chunks_inserted": 0, "chunks_total": 0}

    for corpus_slug in selected:
        try:
            stats = asyncio.run(
                ingest_corpus(
                    corpus_id_str=corpus_slug,
                    force_reingest=force_reingest,
                    dev_mode=dev_mode,
                    max_documents=max_documents,
                )
            )
            total_stats["documents_processed"] += stats["documents_processed"]
            total_stats["chunks_inserted"] += stats["chunks_inserted"]
            total_stats["chunks_total"] += stats["chunks_total"]
        except Exception as exc:
            console.print(f"[red]Corpus {corpus_slug} failed: {exc}[/red]")
            logger.exception("Corpus ingestion failed: %s", corpus_slug)

    # Summary table
    table = Table(title="Ingestion summary", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Documents processed", str(total_stats["documents_processed"]))
    table.add_row("Chunks inserted", str(total_stats["chunks_inserted"]))
    table.add_row("Total chunks", str(total_stats["chunks_total"]))
    console.print(table)


if __name__ == "__main__":
    app()
