"""Corpus loader registry — maps corpus slugs to loader classes."""

from __future__ import annotations

from src.rag.corpus.gdpr_loader import BaseCorpusLoader, CorpusDocument, GDPRLoader

# Minimal stubs for corpora without full loaders yet.
# Replace with real implementations following the GDPRLoader pattern.


class _StubLoader(BaseCorpusLoader):
    """Placeholder for corpora not yet fully implemented."""

    def load_documents(self) -> list[CorpusDocument]:
        return [
            CorpusDocument(
                filename=f"{self.CORPUS_ID}_stub.txt",
                content=(
                    f"{self.CORPUS_NAME}\n\n"
                    f"This corpus is not yet fully loaded. "
                    f"Download the source document and implement a loader following "
                    f"GDPRLoader as the reference pattern."
                ),
            )
        ]


class SOC2Loader(_StubLoader):
    CORPUS_ID = "soc2"
    CORPUS_NAME = "SOC-2 Trust Services Criteria (2022)"
    JURISDICTION = "US"
    VERSION = "2022"
    SOURCE_URL = "https://www.aicpa-cima.com/resources/download/2017-trust-services-criteria"


class ISO27001Loader(_StubLoader):
    CORPUS_ID = "iso27001"
    CORPUS_NAME = "ISO-27001:2022 Annex A"
    JURISDICTION = "global"
    VERSION = "2022"
    SOURCE_URL = None  # ISO standard requires paid license


class SECSPLoader(_StubLoader):
    CORPUS_ID = "sec_sp"
    CORPUS_NAME = "SEC Regulation S-P (17 CFR Part 248)"
    JURISDICTION = "US"
    VERSION = "2024"
    SOURCE_URL = "https://www.ecfr.gov/current/title-17/chapter-II/part-248"


class SECSIDLoader(_StubLoader):
    CORPUS_ID = "sec_sid"
    CORPUS_NAME = "SEC Regulation S-ID (17 CFR Part 248 Subpart C)"
    JURISDICTION = "US"
    VERSION = "2013"
    SOURCE_URL = "https://www.ecfr.gov/current/title-17/chapter-II/part-248/subpart-C"


class CFPBLoader(_StubLoader):
    CORPUS_ID = "cfpb"
    CORPUS_NAME = "CFPB Regulation P (12 CFR Part 1016)"
    JURISDICTION = "US"
    VERSION = "2023"
    SOURCE_URL = "https://www.consumerfinance.gov/rules-policy/regulations/1016/"


class EUAIActLoader(_StubLoader):
    CORPUS_ID = "eu_ai_act"
    CORPUS_NAME = "EU AI Act (Regulation 2024/1689)"
    JURISDICTION = "EU"
    VERSION = "2024-08-01"
    SOURCE_URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=OJ:L_202401689"


# ── Registry ──────────────────────────────────────────────────────────────────

CORPUS_LOADERS: dict[str, type[BaseCorpusLoader]] = {
    "gdpr": GDPRLoader,
    "soc2": SOC2Loader,
    "iso27001": ISO27001Loader,
    "sec_sp": SECSPLoader,
    "sec_sid": SECSIDLoader,
    "cfpb": CFPBLoader,
    "eu_ai_act": EUAIActLoader,
}


def get_loader(corpus_id: str) -> BaseCorpusLoader:
    """Return an instantiated loader for a corpus slug."""
    cls = CORPUS_LOADERS.get(corpus_id)
    if cls is None:
        raise ValueError(f"Unknown corpus: '{corpus_id}'. Available: {list(CORPUS_LOADERS.keys())}")
    return cls()


def list_corpora() -> list[str]:
    """Return all registered corpus slugs."""
    return list(CORPUS_LOADERS.keys())
