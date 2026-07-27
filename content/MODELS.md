# Models — Meridian

This document is a model registry: every AI model used in the pipeline, the task it performs, why it was chosen over alternatives, its known limitations, and the wrapper that exposes it in `src/models/`.

---

## Overview

| Model | HF task | Agent | Wrapper |
|---|---|---|---|
| `deepset/roberta-base-squad2` | Document question answering | Doc agent | `src/models/qa.py` |
| `dslim/bert-base-NER` | Token classification | Doc agent | `src/models/ner.py` |
| `facebook/bart-large-mnli` | Zero-shot classification | Router + doc agent | `src/models/classifier.py` |
| `facebook/bart-large-cnn` | Summarization | Doc agent | `src/models/summarizer.py` |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Text ranking | RAG reranker | `src/models/reranker.py` |
| `sentence-transformers/all-MiniLM-L6-v2` | Sentence similarity | Hallucination gate | `src/models/similarity.py` |
| `text-embedding-3-small` (OpenAI) | Feature extraction | RAG ingestion + retrieval | `src/models/embedder.py` |
| `openai/whisper-large-v3` | Automatic speech recognition | Audio agent | `src/models/asr.py` |
| `pyannote/speaker-diarization-3.1` | Audio classification | Audio agent | `src/models/diarizer.py` |
| `Salesforce/blip2-opt-2.7b` | Image-text-to-text | Vision agent | `src/models/captioner.py` |
| `dandelin/vilt-b32-finetuned-vqa` | Visual question answering | Vision agent | `src/models/vqa.py` |
| `vidore/colpali-v1.2` | Visual document retrieval | Vision agent | `src/models/colpali.py` |
| `google/tapas-base-finetuned-wtq` | Table question answering | Data agent | `src/models/tapas.py` |
| `amazon/chronos-t5-small` | Time series forecasting | Data agent | `src/models/forecaster.py` |
| `Claude Sonnet` (Anthropic) | Text generation | Synthesis agent | `src/models/llm.py` |

All HuggingFace models are called via the HF Inference API unless noted as locally loaded. This avoids GPU requirements in development and keeps the runtime image small.

---

## NLP models

### `deepset/roberta-base-squad2`

**Task:** Document question answering  
**HF page:** https://huggingface.co/deepset/roberta-base-squad2  
**Agent:** Doc agent  
**Called via:** HF Inference API

**What it does:** Given a context passage (a regulatory chunk) and a specific question (e.g., "What is the maximum data retention period stated?"), returns the span of text in the context that best answers the question.

**Why this model:** RoBERTa-base fine-tuned on SQuAD2 is the standard benchmark reference for extractive QA. It handles out-of-scope questions correctly (returns no answer rather than hallucinating) because SQuAD2 includes unanswerable examples. `bert-large-uncased-whole-word-masking-finetuned-squad` was evaluated but showed no meaningful accuracy improvement on the regulatory domain test set and is 3× larger.

**How it's used in Meridian:** For each regulatory entity extracted by the NER model (e.g., a mention of "data retention"), the doc agent constructs a targeted question and runs it against the most relevant retrieved chunks. This gives exact span answers with character-level offsets, which feed directly into the citation system.

**Limitations:**
- Context window limited to 512 tokens; long chunks must be split and re-merged
- Extractive only — can only return spans that exist verbatim in the context, not synthesized answers
- No multilingual support; English only

**Configuration:**

```python
# src/models/qa.py
QA_MODEL_ID = "deepset/roberta-base-squad2"
QA_MAX_CONTEXT_LENGTH = 512       # tokens
QA_SCORE_THRESHOLD = 0.1          # minimum confidence to return an answer
```

---

### `dslim/bert-base-NER`

**Task:** Token classification (named entity recognition)  
**HF page:** https://huggingface.co/dslim/bert-base-NER  
**Agent:** Doc agent  
**Called via:** HF Inference API

**What it does:** Tags each token in the input as one of: PER (person), ORG (organization), LOC (location), MISC (miscellaneous), or O (no entity). Returns entities as spans with start/end character offsets and confidence scores.

**Why this model:** BERT-base-NER is the most widely validated English NER model on CoNLL-2003 and is appropriate for identifying the general entity types that appear in compliance documents (organization names, person names like "Data Protection Officer", dates). Larger models (`bert-large-NER`, `flair/ner-english-ontonotes-large`) were evaluated and showed marginal accuracy gains that did not justify the inference cost.

**How it's used in Meridian:** First-pass NER over submitted policy text. General entities (ORG, MISC) seed the second-pass regulatory NER, which uses zero-shot classification to identify domain-specific categories (`RETENTION_PERIOD`, `LAWFUL_BASIS`, etc.).

**Limitations:**
- Does not recognize regulatory-specific entities out of the box — this is handled by the second NER pass
- 512-token context limit; documents are processed in sliding-window chunks with overlap
- Occasionally mis-classifies multi-word regulatory terms (e.g., "legitimate interest" tagged as MISC rather than a regulatory concept)

---

### `facebook/bart-large-mnli`

**Task:** Zero-shot classification  
**HF page:** https://huggingface.co/facebook/bart-large-mnli  
**Agent:** Input router; doc agent (regulatory entity NER stage 2)  
**Called via:** HF Inference API

**What it does:** Given a sequence and a list of candidate labels, returns a probability distribution over the labels using natural language inference (NLI). Requires no fine-tuning for new label sets.

**Why this model:** BART-large-MNLI is the standard reference for zero-shot classification via NLI. It significantly outperforms `facebook/bart-base-mnli` on short multi-label classification tasks with regulatory terminology. `cross-encoder/nli-deberta-v3-large` was tested and produced higher accuracy but at 3× the inference latency.

**How it's used in Meridian:**

*Input routing:* classifies each uploaded file into a modality (`document`, `audio`, `image`, `tabular`), using the candidate labels as routing signals when the MIME type alone is ambiguous (e.g., a PDF containing only images vs. a text PDF).

*Regulatory NER:* for each candidate span identified by the general NER model, zero-shot classification assigns a regulatory entity type from the domain-specific label set: `RETENTION_PERIOD`, `DATA_SUBJECT_CATEGORY`, `CONSENT_MECHANISM`, `DPO_MENTION`, `LAWFUL_BASIS`, `THIRD_PARTY_TRANSFER`, `SECURITY_MEASURE`.

**Limitations:**
- Inference is slower than a fine-tuned classifier (~150ms per call on HF Inference API)
- Accuracy drops for ambiguous short spans (fewer than 5 tokens)

---

### `facebook/bart-large-cnn`

**Task:** Summarization  
**HF page:** https://huggingface.co/facebook/bart-large-cnn  
**Agent:** Doc agent; audio agent  
**Called via:** HF Inference API

**What it does:** Abstractive summarization — generates a fluent summary that may include phrasing not present verbatim in the source text.

**Why this model:** BART-large-CNN is trained on CNN/DailyMail news articles, which are structurally similar to the dense, factual prose of regulatory documents and compliance policies. `t5-large` was evaluated but produced more abstractive (less faithful) summaries on regulatory text. `sshleifer/distilbart-cnn-12-6` is 40% faster with ~3% lower ROUGE scores — appropriate as a cost-saving swap for high-volume deployments.

**How it's used in Meridian:** Documents over 10,000 tokens are summarized before being passed to the synthesis agent (full text exceeds Claude's context window budget for the synthesis step). Audio transcripts are summarized to extract compliance-relevant statements before downstream RAG retrieval.

**Limitations:**
- Max input length: 1024 tokens. Longer documents require chunk-and-summarize with sliding window
- Occasionally "hallucinates" plausible-sounding but inaccurate legal paraphrases — output must not be used as a citation; only the extractive QA output is cited

---

### `cross-encoder/ms-marco-MiniLM-L-6-v2`

**Task:** Text ranking (cross-encoder reranking)  
**HF page:** https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2  
**Agent:** RAG retrieval pipeline  
**Called via:** HF Inference API

**What it does:** Given a (query, passage) pair, returns a relevance score (not a probability). Unlike bi-encoders (which embed query and passage independently), the cross-encoder reads both together and can model fine-grained interactions.

**Why this model:** Cross-encoders consistently outperform bi-encoders as rerankers because they have full attention between query and passage. MiniLM-L-6 was chosen over MiniLM-L-12 for the original release for its 2× speed advantage; v1.0.3 upgraded to MiniLM-L-12 after benchmarking showed a meaningful precision gain on the GDPR golden dataset.

**How it's used in Meridian:** The RRF-fused top-20 candidate chunks are each scored against the query; the top-5 scoring chunks are passed to the LLM for synthesis.

**Limitations:**
- Quadratic complexity in candidate set size — keeping the candidate pool at ≤ 20 is essential for latency
- Trained on MS-MARCO (web search passages), not regulatory text — may underrank dense legal prose relative to its actual relevance

---

### `sentence-transformers/all-MiniLM-L6-v2`

**Task:** Sentence similarity  
**HF page:** https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2  
**Agent:** Hallucination gate  
**Called via:** Local (loaded in-process for low latency)

**What it does:** Encodes sentences into dense vectors; cosine similarity between vectors measures semantic similarity.

**Why loaded locally:** The hallucination gate runs after synthesis and must not add significant latency. The HF Inference API adds ~100ms round-trip; locally loaded, MiniLM-L6 runs in ~5ms on CPU. Its model size (80 MB) is small enough to include in the Docker image.

**How it's used in Meridian:** For each claim in the synthesis output, the claim text and its cited source chunk are encoded; cosine similarity must exceed `GROUNDEDNESS_THRESHOLD` (default 0.80) for the claim to pass the gate.

---

## Audio models

### `openai/whisper-large-v3`

**Task:** Automatic speech recognition  
**HF page:** https://huggingface.co/openai/whisper-large-v3  
**Agent:** Audio agent  
**Called via:** HF Inference API

**What it does:** End-to-end transformer ASR model. Transcribes audio to text with timestamps; supports 99 languages.

**Why this model:** Whisper large-v3 is the state-of-the-art general-purpose ASR model as of early 2026. It outperforms all alternatives on regulatory domain audio (meetings, conference calls) due to its diverse training data. `whisper-medium` was evaluated as a cost-saving alternative but produced unacceptable WER on domain-specific terminology ("CFPB," "SOC-2," "Regulation S-P").

**Limitations:**
- 30-second context window; longer audio requires chunking with overlap
- Occasional hallucinations on silence or background noise (produces plausible-sounding filler text)
- Speaker labels are not provided — diarization is a separate model

---

### `pyannote/speaker-diarization-3.1`

**Task:** Audio classification (speaker diarization)  
**HF page:** https://huggingface.co/pyannote/speaker-diarization-3.1  
**Agent:** Audio agent  
**Called via:** HF Inference API (requires HF token with model access agreement)

**What it does:** Identifies "who spoke when" — segments audio by speaker and labels each segment.

**Status in Meridian:** Partially integrated (v1.0). Diarization output is annotated onto the Whisper transcript but speaker labels are not yet surfaced in the compliance gap report. Full integration tracked in TODO.md.

**Why this model:** pyannote v3.1 is the leading open-source diarization model. Access requires accepting the model's usage conditions on HuggingFace.

---

## Vision models

### `Salesforce/blip2-opt-2.7b`

**Task:** Image-text-to-text (image captioning)  
**HF page:** https://huggingface.co/Salesforce/blip2-opt-2.7b  
**Agent:** Vision agent  
**Called via:** HF Inference API

**What it does:** Generates a free-form natural language description of an image.

**Why this model:** BLIP-2 produces significantly more detailed captions than BLIP-1 for compliance screenshots (e.g., correctly identifying a cookie consent banner's structure). Evaluated against `Salesforce/blip-image-captioning-large` (smaller, less detailed) and `llava-1.5-7b` (more capable but 4× inference cost). BLIP-2 with OPT-2.7B hits the best cost/accuracy point for captioning tasks that don't require complex reasoning.

**How it's used in Meridian:** Generates an initial description of any uploaded image, which is then used as context for the VQA model and as a retrieval query against the regulatory corpus.

---

### `dandelin/vilt-b32-finetuned-vqa`

**Task:** Visual question answering  
**HF page:** https://huggingface.co/dandelin/vilt-b32-finetuned-vqa  
**Agent:** Vision agent  
**Called via:** HF Inference API

**What it does:** Answers specific natural language questions about an image.

**How it's used in Meridian:** Compliance-specific questions are posed to screenshots and form images:
- "Is there a clearly labeled reject-all option?"
- "Does this banner show pre-ticked consent boxes?"
- "Is a privacy policy link visible?"
- "What data retention period is shown in this table?"

**Limitations:**
- Answers are drawn from a fixed vocabulary (VQA v2 training); open-ended answers are truncated
- Weak on multi-element UI screenshots — struggles when multiple consent options are visually clustered
- Cannot read fine print reliably; use Claude vision for text-heavy image analysis

---

### `vidore/colpali-v1.2`

**Task:** Visual document retrieval  
**HF page:** https://huggingface.co/vidore/colpali-v1.2  
**Agent:** Vision agent  
**Called via:** Local (ColPali requires the `colpali-engine` library, not available via standard HF Inference API)

**What it does:** Embeds entire document pages as images into a shared vector space with query embeddings, enabling retrieval by visual similarity without OCR. Critical for scanned documents where layout conveys meaning.

**Why this model:** Standard text-extraction-then-embed approaches fail on scanned regulatory forms (e.g., a checkbox-based consent form) because the spatial relationship between a field label and its checkbox is lost when text is extracted linearly. ColPali preserves visual layout in its embeddings. ColPali v1.2 outperforms v1.1 by ~4 points on the ViDoRe benchmark.

**Status in Meridian:** The model wrapper and regulatory page image index are under active development (tracked in TODO.md). Currently, scanned PDFs fall back to Claude vision for analysis.

**Limitations:**
- Requires rendering PDF pages to images with `pdf2image` + `poppler`
- Image embeddings are larger (128-dimensional patch embeddings per page) than text embeddings; index is stored separately from the text chunk index
- Inference is GPU-accelerated in production; CPU inference is feasible but slow (~3s/page)

---

## Tabular models

### `google/tapas-base-finetuned-wtq`

**Task:** Table question answering  
**HF page:** https://huggingface.co/google/tapas-base-finetuned-wtq  
**Agent:** Data agent  
**Called via:** HF Inference API

**What it does:** Answers natural language questions about HTML or DataFrame-formatted tables by selecting cells that aggregate to the answer (e.g., sum, count, max).

**Why this model:** TAPAS is the standard reference for neural table QA without SQL. Fine-tuned on WikiTableQuestions (WTQ), which contains aggregation-heavy questions similar to those asked of audit log tables.

**Limitations:**
- Context length limit: the table must fit within 512 tokens; large tables must be pre-filtered to relevant columns
- Cannot reason across multiple tables; each TAPAS call targets a single table
- Aggregate-only — returns cell references rather than prose answers; post-processing converts to natural language

---

### `amazon/chronos-t5-small`

**Task:** Time series forecasting  
**HF page:** https://huggingface.co/amazon/chronos-t5-small  
**Agent:** Data agent  
**Called via:** Local (the `chronos-forecasting` library provides the inference API)

**What it does:** Zero-shot probabilistic time series forecasting. Given a historical sequence of numeric values, returns a distribution over future values (median forecast + 10th/90th percentile bounds).

**Why this model:** Chronos-T5 is the first publicly available foundation model for time series that works zero-shot — no fine-tuning on domain data required. This is essential for the compliance use case where labeled time series training data is not available. Evaluated against `prophet` (Facebook) and `statsforecast` (AutoARIMA); Chronos-T5 produces more calibrated uncertainty estimates and handles irregularly sampled data better.

**How it's used in Meridian:** When the data agent detects a time-indexed column (date + numeric metric), Chronos-T5 forecasts the next 7 periods and flags deviations between actual values and the forecast as potential anomalies. Example: a sharp spike in privileged access events on days not predicted by the historical trend.

**Limitations:**
- Small model (46M parameters) chosen for CPU inference speed; `chronos-t5-large` (710M) is significantly more accurate but adds ~15s to data agent runtime
- Not designed for very short sequences (fewer than 10 historical points); falls back to statistical baseline in this case

---

## Primary LLM

### Claude Sonnet (Anthropic)

**Task:** Text generation (synthesis reasoning)  
**Model string:** `claude-sonnet-4-6`  
**Agent:** Synthesis agent  
**Called via:** Anthropic Python SDK

**What it does:** Receives the merged outputs from all specialist agents and the retrieved regulatory chunks, and reasons over the complete context to produce structured compliance gap findings.

**Why Claude Sonnet:** The synthesis task requires: (a) long-context reasoning over 5–10 retrieved chunks plus agent extraction outputs, (b) structured output conforming to the `ComplianceReport` Pydantic schema, and (c) citations grounded in the provided context rather than parametric memory. Claude Sonnet's 200K context window, instruction following on structured output, and strong grounding behavior make it the best available model for this specific task. GPT-4o was evaluated as an alternative; Claude Sonnet produced fewer hallucinated citations on the groundedness eval set.

**How it's used in Meridian:** Called once per job in the synthesis node with a structured system prompt that enforces: explicit citation of regulatory articles, identification of the specific policy section that is deficient, severity classification with justification, and a remediation recommendation. Output is parsed with a Pydantic model; parsing failures trigger a retry with a clarification instruction.

---

## Model versioning

All model identifiers are defined as constants in `src/models/registry.py`:

```python
# src/models/registry.py

MODELS = {
    "qa":           "deepset/roberta-base-squad2",
    "ner":          "dslim/bert-base-NER",
    "classifier":   "facebook/bart-large-mnli",
    "summarizer":   "facebook/bart-large-cnn",
    "reranker":     "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "similarity":   "sentence-transformers/all-MiniLM-L6-v2",
    "embedder":     "text-embedding-3-small",
    "asr":          "openai/whisper-large-v3",
    "diarizer":     "pyannote/speaker-diarization-3.1",
    "captioner":    "Salesforce/blip2-opt-2.7b",
    "vqa":          "dandelin/vilt-b32-finetuned-vqa",
    "colpali":      "vidore/colpali-v1.2",
    "tapas":        "google/tapas-base-finetuned-wtq",
    "forecaster":   "amazon/chronos-t5-small",
    "llm":          "claude-sonnet-4-6",
}
```

Every compliance report's appendix includes the model version used for each component at the time of job processing, creating a reproducible audit trail.
