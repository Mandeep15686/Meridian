MIT License

Copyright (c) 2026 Meridian Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Third-party notices

Meridian integrates several third-party libraries and services. Their licenses and terms apply to their respective components.

### Open-source dependencies (selected)

| Package | License | Notes |
|---|---|---|
| LangChain / LangGraph | MIT | Orchestration framework |
| LlamaIndex | MIT | RAG framework |
| FastAPI | MIT | Web framework |
| Celery | BSD 3-Clause | Task queue |
| SQLAlchemy | MIT | ORM and connection pooling |
| pgvector (Python client) | MIT | Vector similarity search |
| sentence-transformers | Apache 2.0 | Sentence embeddings |
| transformers (HuggingFace) | Apache 2.0 | Model inference |
| WeasyPrint | BSD 3-Clause | PDF generation |
| pandas | BSD 3-Clause | Tabular data processing |
| RAGAS | Apache 2.0 | RAG evaluation |
| MLflow | Apache 2.0 | Experiment tracking |
| pydantic | MIT | Data validation |
| alembic | MIT | Database migrations |
| pytest | MIT | Testing framework |

### Third-party services

| Service | Terms |
|---|---|
| Anthropic API | https://www.anthropic.com/legal/aup |
| HuggingFace Inference API | https://huggingface.co/terms-of-service |
| LangSmith | https://smith.langchain.com/terms |
| Pinecone | https://www.pinecone.io/terms/ |
| GCP (Cloud Run, Cloud SQL, etc.) | https://cloud.google.com/terms |
| Weights & Biases | https://wandb.ai/site/terms |

### Regulatory corpus sources

The regulatory text ingested into Meridian's corpus is sourced from public domain and freely available government publications:

- **GDPR** — EUR-Lex official publication; reproduction permitted under EUR-Lex reuse policy
- **SEC rules (EDGAR)** — US federal government works; public domain under 17 U.S.C. § 105
- **CFPB regulations** — US federal government works; public domain
- **ISO-27001** — Public interpretive summaries only (ISO standard text is copyright protected); ISO full text requires a licensed copy

Users who provide their own copies of licensed regulatory texts (e.g., ISO standards) are responsible for compliance with the terms of their licenses.
