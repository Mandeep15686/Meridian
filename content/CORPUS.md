# Corpus — Meridian

The regulatory corpus is the knowledge base Meridian retrieves from during compliance analysis. This document describes each corpus, its source, ingestion process, update cadence, and chunk statistics.

---

## Overview

| Corpus ID | Full name | Jurisdiction | Chunks | Source | Last refreshed |
|---|---|---|---|---|---|
| `gdpr` | General Data Protection Regulation | EU | 2,847 | EUR-Lex | 2026-04-01 |
| `soc2` | SOC-2 Trust Services Criteria (2022) | US | 1,203 | AICPA | 2026-03-15 |
| `iso27001` | ISO-27001:2022 Annex A | Global | 734 | Public interpretive text | 2026-03-15 |
| `sec_sp` | SEC Regulation S-P | US | 892 | EDGAR | 2026-04-01 |
| `cfpb` | CFPB Regulation P | US | 1,847 | CFPB | 2026-04-07 |
| `sec_sid` | SEC Regulation S-ID | US | 943 | EDGAR | 2026-04-07 |
| **Total** | | | **8,466** | | |

---

## Corpus detail

### GDPR (`gdpr`)

**Source:** EUR-Lex official GDPR text  
**URL:** https://eur-lex.europa.eu/eli/reg/2016/679/oj  
**Coverage:** All 99 articles and 173 recitals  
**Language:** English (bilingual ingestion planned for v1.1)  
**Chunks:** 2,847  
**Update cadence:** Quarterly; EUR-Lex publishes consolidated amendments  

**Ingestion notes:**
- Recitals are ingested separately from articles and tagged with `article_type: recital` to avoid them being cited as binding requirements
- Each article's sub-paragraphs are chunked as individual units to preserve the clause-level granularity needed for precise citations (e.g., "Article 6(1)(f)" rather than "Article 6")
- Cross-references between articles are preserved in the `section_path` metadata field

**Key articles for compliance gap detection:**
- Articles 5–11: Lawfulness of processing, consent, conditions for consent
- Articles 12–14: Transparency and information obligations (most commonly cited in enforcement actions)
- Articles 15–22: Data subject rights
- Articles 24–32: Controller and processor obligations
- Articles 33–34: Breach notification
- Articles 35–36: Data protection impact assessment

---

### SOC-2 Trust Services Criteria (`soc2`)

**Source:** AICPA Trust Services Criteria (2022 revision)  
**URL:** https://www.aicpa-cima.com/resources/download/2017-trust-services-criteria-with-2022-points-of-focus-updates  
**Coverage:** All five Trust Service Categories (Security, Availability, Processing Integrity, Confidentiality, Privacy) + Common Criteria  
**Chunks:** 1,203  
**Update cadence:** AICPA revises approximately every 3–5 years; check annually  

**Ingestion notes:**
- The AICPA document is behind a registration wall; a downloaded copy is stored in `data/corpus_sources/soc2_tsc_2022.pdf` (not committed to the repo; see `scripts/corpus_sources/README.md` for acquisition instructions)
- Points of Focus (non-binding examples) are tagged separately from the binding criteria text
- Each criterion (e.g., CC6.1) is chunked as a single unit to preserve its complete scope

---

### ISO-27001 Annex A (`iso27001`)

**Source:** Public interpretive summaries (ISO full text requires a paid license)  
**Coverage:** All 93 Annex A controls in ISO-27001:2022  
**Chunks:** 734  
**Update cadence:** ISO revised the standard in 2022; next revision expected ~2027  

**Ingestion notes:**
- The ISO-27001 full standard text is not publicly available; Meridian uses authoritative public interpretive summaries from ISMS.online and BSI as proxies
- Citations reference the control ID (e.g., "A.8.1.1") and control name, not page numbers
- Users who require certified ISO-27001 text can provide their own purchased copy via the custom corpus upload feature (planned for v2.0)

---

### SEC Regulation S-P (`sec_sp`)

**Source:** EDGAR Electronic Code of Federal Regulations  
**URL:** https://www.ecfr.gov/current/title-17/chapter-II/part-248  
**Coverage:** 17 CFR Part 248 (Regulation S-P: Privacy of Consumer Financial Information)  
**Chunks:** 892  
**Update cadence:** Monthly check via EDGAR API; the rule was significantly amended in 2024  

**Ingestion notes:**
- EDGAR provides the full regulatory text via a public API; no authentication required (but the `EDGAR_USER_AGENT` environment variable must be set per EDGAR's fair use policy)
- SEC enforcement actions are downloaded separately and stored in `data/golden/` as part of the gap detection golden dataset construction
- The 2024 amendments to Reg S-P (expanded cybersecurity incident notification requirements) are included and tagged with `effective_date: 2024-08-02`

---

### CFPB Regulation P (`cfpb`)

**Source:** Consumer Financial Protection Bureau public API  
**URL:** https://www.consumerfinance.gov/rules-policy/regulations/  
**Coverage:** 12 CFR Part 1016 (Privacy of Consumer Financial Information)  
**Chunks:** 1,847  
**Update cadence:** CFPB updates regulations periodically; automated monthly refresh  

**Ingestion notes:**
- CFPB provides a machine-readable JSON API for regulation text; the ingestion script uses this directly rather than scraping HTML
- Staff commentary (interpretive guidance) is ingested alongside the regulation text and tagged as `document_type: commentary`

---

## Adding a new corpus

To add a new regulatory framework to Meridian, follow these steps:

### 1. Create a corpus loader

Create a new file in `src/rag/corpus/`:

```python
# src/rag/corpus/hipaa_loader.py

from src.rag.corpus.base import BaseCorpusLoader, CorpusDocument

class HIPAALoader(BaseCorpusLoader):
    CORPUS_ID = "hipaa"
    CORPUS_NAME = "HIPAA Privacy Rule"
    JURISDICTION = "US"
    VERSION = "2013"
    SOURCE_URL = "https://www.hhs.gov/hipaa/for-professionals/privacy/laws-regulations/"

    def load_documents(self) -> list[CorpusDocument]:
        """Fetch and return raw documents from the source."""
        # Implement: download or read local files, return CorpusDocument objects
        ...

    def get_article_metadata(self, section_text: str) -> dict:
        """Extract article/section metadata from a chunk of text."""
        # Implement: parse section headers to extract CFR citation
        ...
```

### 2. Register the loader

```python
# src/rag/corpus/registry.py

from src.rag.corpus.hipaa_loader import HIPAALoader

CORPUS_LOADERS = {
    "gdpr":      GDPRLoader,
    "soc2":      SOC2Loader,
    "iso27001":  ISO27001Loader,
    "sec_sp":    SECSPLoader,
    "cfpb":      CFPBLoader,
    "hipaa":     HIPAALoader,   # <-- add here
}
```

### 3. Run ingestion

```bash
python scripts/ingest_corpus.py --source hipaa
```

### 4. Add to `CORPUS_SOURCES` in `.env`

```env
CORPUS_SOURCES=gdpr,soc2,iso27001,sec_sp,cfpb,hipaa
```

### 5. Update golden datasets

Add 20–30 Q&A pairs for the new corpus to `data/golden/` and update the gap detection dataset with at least 15 examples from the new framework.

---

## Corpus freshness and updates

### Automated freshness check

The Celery Beat scheduler runs a corpus freshness check daily at 2am UTC:

```python
# scripts/check_corpus_freshness.py
# Alerts if any corpus has not been refreshed within CORPUS_MAX_STALENESS_DAYS (default 30)
```

The `GET /v1/corpus/status` endpoint exposes freshness status for monitoring dashboards.

### Incremental ingestion

The ingestion script is idempotent: it hashes each source document and only re-chunks and re-embeds documents that have changed since the last ingestion. This makes incremental updates fast:

```bash
# Full re-ingestion of all corpora (first run or after EMBEDDING_MODEL change)
python scripts/ingest_corpus.py --source all --force-reingest

# Incremental update (checks for new/changed documents only)
python scripts/ingest_corpus.py --source gdpr,sec_sp

# Dev mode: ingest only first 500 chunks per corpus (fast, for local dev)
python scripts/ingest_corpus.py --source gdpr --dev-mode
```

### After changing the embedding model

If `EMBEDDING_MODEL` is changed in configuration, all existing embeddings are incompatible with new query embeddings. A full re-ingestion is required:

```bash
# Re-ingest everything (expect ~2 hours for the full corpus)
python scripts/ingest_corpus.py --source all --force-reingest

# Rebuild the IVFFlat index after re-ingestion
psql $DATABASE_URL -c "REINDEX INDEX CONCURRENTLY chunks_embedding_idx;"
```

---

## Corpus quality signals

Run the corpus diagnostic to check chunk quality and coverage:

```bash
python scripts/diagnose_corpus.py
```

Output includes:
- Chunk count and average token count per corpus
- Percentage of chunks with complete metadata (article, jurisdiction, effective_date)
- Sample of 5 chunks per corpus for manual review
- Embedding coverage (% of chunks with non-null embeddings)
- IVFFlat index status and last rebuild timestamp
