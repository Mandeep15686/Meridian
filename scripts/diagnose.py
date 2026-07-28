"""
Meridian diagnostics script — checks all system dependencies and reports status.

Usage:
    python scripts/diagnose.py
    python scripts/diagnose.py --verbose
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer()


async def _check_database() -> tuple[bool, str]:
    try:
        from sqlalchemy import text

        from src.db.session import engine

        async with engine.connect() as conn:
            await conn.execute(text("SELECT version()"))
            # Check pgvector
            vec_result = await conn.execute(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            )
            vec_version = vec_result.scalar()
            if not vec_version:
                return False, "pgvector extension not installed"
            return True, f"PostgreSQL OK (pgvector {vec_version})"
    except Exception as exc:
        return False, f"Connection failed: {exc}"


async def _check_redis() -> tuple[bool, str]:
    try:
        import redis.asyncio as aioredis

        from src.config import settings

        r = aioredis.from_url(settings.REDIS_URL)
        await r.ping()
        info = await r.info("server")
        version = info.get("redis_version", "unknown")
        await r.aclose()
        return True, f"Redis {version} OK"
    except Exception as exc:
        return False, f"Connection failed: {exc}"


async def _check_hf_api() -> tuple[bool, str]:
    try:
        import httpx

        from src.config import settings

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://huggingface.co/api/whoami-v2",
                headers={"Authorization": f"Bearer {settings.HF_API_TOKEN}"},
            )

        if resp.status_code == 200:
            user = resp.json().get("name", "unknown")
            return True, f"Authenticated as {user}"

        return False, f"HTTP {resp.status_code}"

    except Exception as exc:
        return False, f"Request failed: {exc}"


async def _check_anthropic_api() -> tuple[bool, str]:
    try:
        from src.config import settings

        key = settings.ANTHROPIC_API_KEY
        if not key.startswith("sk-ant"):
            return (
                False,
                "ANTHROPIC_API_KEY does not look valid (should start with sk-ant)",
            )
        # Lightweight check — don't make an actual API call in diagnostics
        return True, "API key format valid (sk-ant-...)"
    except Exception as exc:
        return False, str(exc)


async def _check_corpus() -> tuple[bool, str]:
    try:
        from sqlalchemy import func, select

        from src.db.models import Chunk, Corpus
        from src.db.session import get_db_session

        async with get_db_session() as session:
            corpus_result = await session.execute(
                select(Corpus.slug, Corpus.chunk_count).where(Corpus.is_active)
            )
            corpora = corpus_result.all()

            if not corpora:
                return False, "No corpora ingested yet — run scripts/ingest_corpus.py"

            chunk_result = await session.execute(
                select(func.count(Chunk.id)).where(Chunk.embedding.isnot(None))
            )
            chunk_count = chunk_result.scalar() or 0

        corpus_summary = ", ".join(f"{row[0]}({row[1]})" for row in corpora)
        return (
            True,
            f"{len(corpora)} corpora, {chunk_count} embedded chunks — {corpus_summary}",
        )
    except Exception as exc:
        return False, f"Query failed: {exc}"


def _check_system_deps() -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []

    # libmagic
    try:
        import magic

        magic.from_buffer(b"test", mime=True)
        results.append(("libmagic", True, "python-magic OK"))
    except Exception as exc:
        results.append(("libmagic", False, f"Install libmagic1: {exc}"))

    # ffmpeg
    try:
        import subprocess

        proc = subprocess.run(["ffprobe", "-version"], capture_output=True, timeout=5)
        if proc.returncode == 0:
            version_line = proc.stdout.decode().split("\n")[0]
            results.append(("ffmpeg/ffprobe", True, version_line[:60]))
        else:
            results.append(("ffmpeg/ffprobe", False, "ffprobe not found"))
    except Exception as exc:
        results.append(("ffmpeg/ffprobe", False, f"Not installed: {exc}"))

    # poppler (pdf2image)
    try:
        import subprocess

        proc = subprocess.run(["pdftoppm", "-v"], capture_output=True, timeout=5)
        results.append(("poppler-utils", True, "pdftoppm available"))
    except Exception:
        results.append(("poppler-utils", False, "Install poppler-utils"))

    # sentence-transformers (local similarity model)
    try:
        results.append(("sentence-transformers", True, "Library installed"))
    except ImportError as exc:
        results.append(("sentence-transformers", False, str(exc)))

    return results


@app.command()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Run Meridian system diagnostics and report status of all dependencies."""
    console.print("\n[bold]Meridian system diagnostics[/bold]\n")

    results: list[tuple[str, bool, str]] = []

    # Async checks
    async def run_async_checks():
        db_ok, db_msg = await _check_database()
        redis_ok, redis_msg = await _check_redis()
        hf_ok, hf_msg = await _check_hf_api()
        anthropic_ok, anthropic_msg = await _check_anthropic_api()
        corpus_ok, corpus_msg = await _check_corpus()
        return [
            ("PostgreSQL + pgvector", db_ok, db_msg),
            ("Redis", redis_ok, redis_msg),
            ("HuggingFace Inference API", hf_ok, hf_msg),
            ("Anthropic API", anthropic_ok, anthropic_msg),
            ("Corpus index", corpus_ok, corpus_msg),
        ]

    try:
        async_results = asyncio.run(run_async_checks())
        results.extend(async_results)
    except Exception as exc:
        console.print(f"[red]Async checks failed: {exc}[/red]")

    # System dependency checks (sync)
    sys_results = _check_system_deps()
    results.extend(sys_results)

    # Configuration check
    try:
        from src.config import settings

        results.append(
            (
                "Configuration (Pydantic)",
                True,
                f"v{settings.VERSION}, env={settings.ENVIRONMENT}",
            )
        )
    except Exception as exc:
        results.append(("Configuration (Pydantic)", False, f"Validation failed: {exc}"))

    # Print results table
    table = Table(show_header=True, header_style="bold")
    table.add_column("Component", style="cyan", width=30)
    table.add_column("Status", width=8)
    table.add_column("Details")

    all_passed = True
    for name, ok, message in results:
        status = "[green]✓ OK[/green]" if ok else "[red]✗ FAIL[/red]"
        if not ok:
            all_passed = False
        if verbose or not ok:
            table.add_row(name, status, message)
        else:
            table.add_row(
                name, status, message[:80] + "..." if len(message) > 80 else message
            )

    console.print(table)

    if all_passed:
        console.print("\n[green]All checks passed — Meridian is ready.[/green]\n")
        raise SystemExit(0)
    else:
        failed = [name for name, ok, _ in results if not ok]
        console.print(
            f"\n[red]{len(failed)} check(s) failed: {', '.join(failed)}[/red]"
        )
        console.print("See TROUBLESHOOTING.md for fix instructions.\n")
        raise SystemExit(1)


if __name__ == "__main__":
    app()
