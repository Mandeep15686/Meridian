"""Base corpus loader interface and GDPR implementation."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Base types ────────────────────────────────────────────────────────────────

@dataclass
class CorpusDocument:
    """A single source document to be ingested."""
    filename: str
    content: str
    source_url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkMetadata:
    """Article-level metadata for a text chunk."""
    article: str | None = None
    article_title: str | None = None
    section_path: list[str] = field(default_factory=list)


class BaseCorpusLoader(ABC):
    """Abstract base class for all regulatory corpus loaders."""

    # Subclasses must define these
    CORPUS_ID: str = ""
    CORPUS_NAME: str = ""
    JURISDICTION: str = ""
    VERSION: str = ""
    SOURCE_URL: str | None = None

    @abstractmethod
    def load_documents(self) -> list[CorpusDocument]:
        """Fetch and return raw documents from the source."""
        ...

    def get_chunk_metadata(self, chunk_text: str) -> ChunkMetadata:
        """
        Extract article/section metadata from a chunk of text.

        Override in subclasses to provide regulation-specific parsing.
        Default returns empty metadata.
        """
        return ChunkMetadata()

    def corpus_info(self) -> dict[str, str]:
        """Return corpus metadata dict for database insertion."""
        return {
            "slug": self.CORPUS_ID,
            "name": self.CORPUS_NAME,
            "jurisdiction": self.JURISDICTION,
            "version": self.VERSION,
            "source_url": self.SOURCE_URL,
        }


# ── GDPR corpus loader ────────────────────────────────────────────────────────

GDPR_ARTICLE_RE = re.compile(
    r"Article\s+(\d+[a-z]?)\s*[–\-]\s*(.+?)(?=\n|Article|\Z)",
    re.IGNORECASE,
)
GDPR_RECITAL_RE = re.compile(r"Recital\s+(\d+)", re.IGNORECASE)


class GDPRLoader(BaseCorpusLoader):
    """
    GDPR corpus loader — fetches the full regulation text from EUR-Lex.

    In production, this downloads the official GDPR HTML from EUR-Lex and
    parses it into per-article documents. For offline/dev use, falls back
    to a locally cached copy in data/corpus_sources/gdpr_en.txt.
    """

    CORPUS_ID = "gdpr"
    CORPUS_NAME = "General Data Protection Regulation (GDPR)"
    JURISDICTION = "EU"
    VERSION = "2018-05-25"
    SOURCE_URL = "https://eur-lex.europa.eu/eli/reg/2016/679/oj"

    LOCAL_CACHE = Path("data/corpus_sources/gdpr_en.txt")
    EURLEX_URL = (
        "https://eur-lex.europa.eu/legal-content/EN/TXT/TEXT/"
        "?uri=CELEX%3A32016R0679"
    )

    def load_documents(self) -> list[CorpusDocument]:
        """Load GDPR text, preferring local cache over network fetch."""
        if self.LOCAL_CACHE.exists():
            logger.info("Loading GDPR from local cache: %s", self.LOCAL_CACHE)
            return self._parse_gdpr_text(self.LOCAL_CACHE.read_text(encoding="utf-8"))

        logger.info("Fetching GDPR text from EUR-Lex...")
        text = self._fetch_from_eurlex()
        if text:
            self.LOCAL_CACHE.parent.mkdir(parents=True, exist_ok=True)
            self.LOCAL_CACHE.write_text(text, encoding="utf-8")
            return self._parse_gdpr_text(text)

        # Fall back to minimal embedded sample for testing
        logger.warning("EUR-Lex fetch failed — using minimal sample text")
        return self._get_sample_documents()

    def _fetch_from_eurlex(self) -> str | None:
        """Download the GDPR full text from EUR-Lex."""
        try:
            import httpx
            from src.config import settings
            response = httpx.get(
                self.EURLEX_URL,
                headers={"User-Agent": settings.EDGAR_USER_AGENT},
                timeout=30.0,
                follow_redirects=True,
            )
            if response.status_code == 200:
                # Strip HTML tags
                from html.parser import HTMLParser

                class _TextExtractor(HTMLParser):
                    def __init__(self):
                        super().__init__()
                        self._text: list[str] = []
                    def handle_data(self, data: str):
                        self._text.append(data)
                    def get_text(self) -> str:
                        return " ".join(self._text)

                parser = _TextExtractor()
                parser.feed(response.text)
                return parser.get_text()
        except Exception as exc:
            logger.warning("EUR-Lex fetch failed: %s", exc)
        return None

    def _parse_gdpr_text(self, text: str) -> list[CorpusDocument]:
        """Split GDPR text into per-article documents."""
        documents: list[CorpusDocument] = []

        # Split on "Article N" boundaries
        parts = re.split(r"(Article\s+\d+[a-z]?\b)", text, flags=re.IGNORECASE)

        current_article: str | None = None
        current_lines: list[str] = []

        for part in parts:
            if re.match(r"Article\s+\d+[a-z]?\b", part, re.IGNORECASE):
                if current_article and current_lines:
                    content = f"{current_article}\n\n" + "\n".join(current_lines)
                    documents.append(CorpusDocument(
                        filename=f"gdpr_{current_article.lower().replace(' ', '_')}.txt",
                        content=content.strip(),
                        source_url=self.SOURCE_URL,
                        metadata={"article": current_article, "regulation": "gdpr"},
                    ))
                current_article = part.strip()
                current_lines = []
            else:
                current_lines.extend(part.split("\n"))

        # Flush last article
        if current_article and current_lines:
            content = f"{current_article}\n\n" + "\n".join(current_lines)
            documents.append(CorpusDocument(
                filename=f"gdpr_{current_article.lower().replace(' ', '_')}.txt",
                content=content.strip(),
                source_url=self.SOURCE_URL,
                metadata={"article": current_article, "regulation": "gdpr"},
            ))

        logger.info("Parsed %d GDPR article documents", len(documents))
        return documents if documents else self._get_sample_documents()

    def get_chunk_metadata(self, chunk_text: str) -> ChunkMetadata:
        """Extract GDPR article number and title from a chunk."""
        # Match "Article 13 — Information to be provided"
        match = re.search(
            r"Article\s+(\d+[a-z]?)(?:\s*[–\-]\s*(.+?))?(?:\n|$)",
            chunk_text[:200],
            re.IGNORECASE,
        )
        if match:
            article_num = match.group(1)
            article_title = (match.group(2) or "").strip()[:128]
            return ChunkMetadata(
                article=f"Article {article_num}",
                article_title=article_title or None,
                section_path=["GDPR", f"Article {article_num}"],
            )
        return ChunkMetadata()

    def _get_sample_documents(self) -> list[CorpusDocument]:
        """Return minimal sample documents for testing without network access."""
        samples = [
            CorpusDocument(
                filename="gdpr_article_5.txt",
                content=(
                    "Article 5 — Principles relating to processing of personal data\n\n"
                    "1. Personal data shall be:\n"
                    "(a) processed lawfully, fairly and in a transparent manner in relation "
                    "to the data subject ('lawfulness, fairness and transparency');\n"
                    "(b) collected for specified, explicit and legitimate purposes and not "
                    "further processed in a manner that is incompatible with those purposes;\n"
                    "(c) adequate, relevant and limited to what is necessary in relation to "
                    "the purposes for which they are processed ('data minimisation');\n"
                    "(d) accurate and, where necessary, kept up to date;\n"
                    "(e) kept in a form which permits identification of data subjects for no "
                    "longer than is necessary for the purposes ('storage limitation');\n"
                    "(f) processed in a manner that ensures appropriate security."
                ),
                source_url=self.SOURCE_URL,
                metadata={"article": "Article 5", "regulation": "gdpr"},
            ),
            CorpusDocument(
                filename="gdpr_article_13.txt",
                content=(
                    "Article 13 — Information to be provided where personal data are collected\n\n"
                    "1. Where personal data relating to a data subject are collected from the "
                    "data subject, the controller shall, at the time when personal data are "
                    "obtained, provide the data subject with the following information:\n"
                    "(a) the identity and the contact details of the controller;\n"
                    "(b) the contact details of the data protection officer, where applicable;\n"
                    "(c) the purposes of the processing for which the personal data are intended "
                    "as well as the legal basis for the processing;\n\n"
                    "2. In addition to the information referred to in paragraph 1, the controller "
                    "shall, at the time when personal data are obtained, provide the data subject "
                    "with the following further information necessary to ensure fair processing:\n"
                    "(a) the period for which the personal data will be stored, or if that is not "
                    "possible, the criteria used to determine that period;\n"
                    "(b) where the processing is based on point (f) of Article 6(1), the "
                    "legitimate interests pursued by the controller or by a third party."
                ),
                source_url=self.SOURCE_URL,
                metadata={"article": "Article 13", "regulation": "gdpr"},
            ),
            CorpusDocument(
                filename="gdpr_article_17.txt",
                content=(
                    "Article 17 — Right to erasure ('right to be forgotten')\n\n"
                    "1. The data subject shall have the right to obtain from the controller "
                    "the erasure of personal data concerning him or her without undue delay "
                    "and the controller shall have the obligation to erase personal data "
                    "without undue delay where one of the following grounds applies:\n"
                    "(a) the personal data are no longer necessary in relation to the purposes "
                    "for which they were collected or otherwise processed;\n"
                    "(b) the data subject withdraws consent on which the processing is based "
                    "and where there is no other legal ground for the processing;\n"
                    "(c) the data subject objects to the processing pursuant to Article 21(1) "
                    "and there are no overriding legitimate grounds for the processing."
                ),
                source_url=self.SOURCE_URL,
                metadata={"article": "Article 17", "regulation": "gdpr"},
            ),
        ]
        return samples
