# Glossary — Meridian

Terms are grouped by domain. Regulatory terms reflect their legal definitions in the relevant frameworks. AI/ML terms reflect their usage within the Meridian codebase specifically.

---

## Regulatory terms

**Article 6 (GDPR)**
The GDPR provision that defines the six lawful bases for processing personal data: consent, contract, legal obligation, vital interests, public task, and legitimate interests. Absence of a stated lawful basis in a privacy policy is one of the most commonly detected compliance gaps.

**Controller**
The entity that determines the purposes and means of processing personal data. In practice, this is typically the company whose policy Meridian is analyzing. Distinct from a *processor*, which processes data on behalf of the controller.

**Data subject**
Any identified or identifiable natural person whose personal data is being processed. GDPR grants data subjects several rights including access, rectification, erasure, and portability.

**DPO (Data Protection Officer)**
An individual appointed by certain organizations under GDPR Article 37 to oversee data protection strategy and compliance. Organizations that process large volumes of sensitive data, or that are public authorities, are required to appoint a DPO. Meridian's NER model specifically identifies DPO mentions as a compliance signal.

**GDPR (General Data Protection Regulation)**
EU Regulation 2016/679, in force since May 25, 2018. The primary data protection regulation that Meridian analyzes policies against. Enforced by national Data Protection Authorities (DPAs) in each EU member state.

**Lawful basis**
One of the six bases under GDPR Article 6 that permits the processing of personal data. A privacy policy that fails to state the specific lawful basis for each processing activity is non-compliant with Article 13(1)(c).

**Legitimate interest**
One of the six GDPR lawful bases (Article 6(1)(f)), allowing processing where it is necessary for the controller's legitimate interests, provided those interests are not overridden by the data subject's rights. Frequently miscited in privacy policies without the required balancing test.

**Privacy notice**
A document provided to data subjects explaining how their personal data is processed. Meridian's primary input document type. Synonyms: privacy policy, privacy statement.

**Processing**
Any operation performed on personal data: collection, storage, use, disclosure, erasure, etc. GDPR's definition is deliberately broad.

**Processor**
An entity that processes personal data on behalf of the controller. Distinct from the controller. GDPR Article 28 requires a written contract (Data Processing Agreement) between controller and processor.

**Regulation S-P**
17 CFR Part 248, the SEC rule requiring broker-dealers and investment advisers to protect customer financial information. The financial services equivalent of GDPR for US-regulated entities.

**Retention period**
The length of time personal data is stored before deletion. GDPR Article 13(2)(a) requires controllers to specify this period (or the criteria for determining it) in the privacy notice. Absence of a specified retention period is one of the most frequently cited GDPR violations.

**Right to erasure**
GDPR Article 17, colloquially the "right to be forgotten." Data subjects may request deletion of their personal data under certain conditions. Privacy policies must inform data subjects of this right per Article 13(2)(b).

**SOC-2 (Service Organization Control 2)**
A US auditing framework developed by the AICPA for evaluating a service organization's controls related to security, availability, processing integrity, confidentiality, and privacy. SOC-2 Type II certifications are required by many enterprise customers.

**Trust Services Criteria (TSC)**
The control requirements that form the basis of a SOC-2 audit. Organized into Common Criteria (CC) and additional criteria for availability, confidentiality, processing integrity, and privacy.

---

## AI / ML terms

**Attention (cross-attention)**
The mechanism by which transformer-based models weigh the relevance of different parts of the input when producing an output. Cross-encoders apply attention *across* both the query and the passage simultaneously, enabling finer-grained relevance scoring than bi-encoders.

**BM25**
Best Match 25 — a probabilistic ranking function used in information retrieval. The standard algorithm underlying most keyword search systems (including Elasticsearch). In Meridian, BM25 is approximated using PostgreSQL's `ts_rank_cd` function and forms the keyword retrieval component of the hybrid search pipeline.

**Bi-encoder**
An embedding model architecture that encodes the query and the document independently into separate vector representations, then computes similarity between the two vectors. Fast for large-scale retrieval (precompute document embeddings) but misses fine-grained query-document interactions. Used in Meridian's dense retrieval stage.

**Chunk**
A contiguous excerpt of a source document, produced by the chunking stage of the ingestion pipeline. The basic unit of retrieval and citation. Chunks in Meridian are produced by semantic chunking and stored in the `chunks` PostgreSQL table with vector embeddings and full-text search indexes.

**Cross-encoder**
A reranking model architecture that reads both the query and a candidate document together in a single forward pass, enabling it to model query-document interactions directly. More accurate than bi-encoders for ranking but too slow for initial retrieval (no precomputation). Used in Meridian's third retrieval stage (reranking).

**CRAG (Corrective RAG)**
A retrieval technique that evaluates the quality of retrieved documents and, if quality is low, falls back to web search or alternate sources. Meridian does not implement full CRAG but the hallucination gate is inspired by its quality-check philosophy.

**Dense retrieval**
Retrieval using vector similarity between dense embedding representations of queries and documents. Captures semantic similarity but may miss exact keyword matches. Meridian uses pgvector for dense retrieval in stage 1.

**Diarization**
The process of segmenting an audio recording by speaker — answering "who spoke when." Implemented in Meridian's audio agent using `pyannote/speaker-diarization-3.1`.

**Embedding**
A dense numerical vector representation of text (or an image, audio clip, etc.) produced by an encoder model. In Meridian, text embeddings are produced by `text-embedding-3-small` and stored in the `chunks.embedding` column.

**F1 score**
The harmonic mean of precision and recall: `2 × (precision × recall) / (precision + recall)`. Used as the primary accuracy metric for the gap detection eval harness. An F1 of 1.0 means perfect precision and recall; F1 of 0 means the system detected no correct gaps.

**Faithfulness (RAGAS)**
A RAGAS metric measuring the fraction of claims in the generated answer that are supported by the retrieved context. A faithfulness score below 0.80 is a hallucination warning signal.

**Groundedness**
In Meridian, a per-claim quality signal measuring cosine similarity between a synthesized claim and its cited source chunk. Claims with groundedness below `GROUNDEDNESS_THRESHOLD` (default 0.80) are rejected by the hallucination gate.

**Hallucination**
A model-generated output that is factually incorrect or unsupported by the provided context. In compliance analysis, a hallucinated regulatory citation (citing an article that doesn't say what the system claims) would be a high-severity error. The hallucination gate is Meridian's primary defense.

**Hallucination gate**
A LangGraph node in Meridian's pipeline that verifies every synthesized claim against its cited source chunk using semantic similarity. Claims that fail the groundedness check trigger a synthesis retry or are marked as uncertain in the final report.

**Hybrid retrieval**
A retrieval strategy combining dense (semantic) and sparse (keyword) retrieval methods. Meridian combines pgvector dense retrieval with PostgreSQL BM25 keyword retrieval, fused via Reciprocal Rank Fusion. Outperforms either method alone on regulatory text.

**IVFFlat**
Inverted File Flat — a vector index type in pgvector that clusters embeddings into `lists` groups and searches only a subset of those groups at query time (approximate nearest neighbor). Fast and memory-efficient but requires periodic rebuilding as the corpus grows.

**LangGraph**
A Python framework from LangChain for building stateful multi-agent workflows as directed graphs. Meridian's core orchestration layer. Key features used: `StateGraph`, `Send` (parallel fan-out), `add_conditional_edges`, `checkpointer` (resumable execution).

**LLM-as-judge**
An evaluation technique where a capable LLM (typically Claude or GPT-4) is used to assess the quality of another model's outputs. In Meridian, the agent judge LLM evaluates LangSmith traces for routing correctness, tool use quality, and citation accuracy.

**NER (Named Entity Recognition)**
A token classification task that identifies and categorizes named entities in text. Meridian uses NER in two stages: general NER (`dslim/bert-base-NER`) followed by regulatory-domain NER (zero-shot classification) to identify compliance-specific entities.

**RAGAS**
Retrieval Augmented Generation Assessment — an open-source evaluation framework that measures RAG pipeline quality using four metrics: faithfulness, answer relevancy, context precision, and context recall. Used as Meridian's primary RAG quality evaluation tool.

**RAG (Retrieval-Augmented Generation)**
A technique for grounding LLM outputs in a specific knowledge base by retrieving relevant documents at inference time and including them in the model's context. Meridian's core mechanism for ensuring compliance analysis is grounded in authoritative regulatory text rather than LLM parametric memory.

**Reciprocal Rank Fusion (RRF)**
A rank aggregation algorithm that combines multiple ranked lists (e.g., dense retrieval results and BM25 results) into a single unified ranking. Score formula: `RRF(d) = Σ 1 / (k + rank(d))` where `k=60` is a constant. Simple, effective, and requires no parameter tuning.

**Reranker**
A cross-encoder model that rescores a small set of candidate documents (top-20 from RRF) to produce a more accurate ranking (top-5) for the LLM. In Meridian: `cross-encoder/ms-marco-MiniLM-L-12-v2`.

**Send (LangGraph)**
A LangGraph API for parallel node execution. The router node emits one `Send` object per detected input modality; LangGraph runs all target nodes concurrently. This is the mechanism for Meridian's parallel specialist agent fan-out.

**Semantic chunking**
A document chunking strategy that splits text at semantic boundaries (natural topic shifts) rather than at fixed token counts. Produces more coherent chunks that are better aligned with retrieval units. Implemented via LlamaIndex `SemanticChunker`.

**Synthesis agent**
The LangGraph node that receives all specialist agent outputs and retrieved regulatory chunks, and uses Claude Sonnet to reason over the combined context to identify compliance gaps. The synthesis agent is the only node that calls Claude directly.

**TypedDict (Python)**
A Python type hint construct that defines a dictionary with a specific set of typed keys. Used in Meridian for `MeridianState` — the shared state object passed between all LangGraph nodes. TypedDict enables static type checking with `mypy` on state access patterns.

**VAD (Voice Activity Detection)**
An algorithm that identifies segments of audio containing speech versus silence or noise. Meridian uses VAD to split audio files at natural silence boundaries before passing segments to Whisper, improving transcription accuracy.

**VQA (Visual Question Answering)**
A multimodal task where a model answers natural language questions about an image. In Meridian's vision agent, VQA (`vilt-b32-finetuned-vqa`) is used to extract structured compliance information from screenshots (e.g., "Is there a reject-all button visible?").

**Zero-shot classification**
A text classification approach where the model classifies inputs into categories it was not explicitly trained on, using natural language descriptions of the categories as labels. Meridian uses `facebook/bart-large-mnli` for zero-shot classification of document types and regulatory entity categories.
