# Evaluation methodology — Meridian

Meridian uses a three-layer evaluation architecture: RAG quality metrics (RAGAS), task-level accuracy (custom F1 harness), and agent behavior scoring (LLM-as-judge). This document explains the design philosophy behind each layer, how the golden datasets were constructed, and how to interpret and act on the results.

---

## Why three evaluation layers

A compliance gap detection pipeline has distinct failure modes at each stage:

| Stage | Failure mode | What measures it |
|---|---|---|
| Retrieval | Wrong regulatory clauses retrieved | RAGAS context precision/recall |
| Synthesis | Correct clauses retrieved but wrong gaps identified | Gap detection F1 |
| Synthesis | Correct gaps identified but claims not grounded | Groundedness score |
| Agents | Wrong routing, wrong tool use, poor reasoning | LLM-as-judge |
| Latency | Pipeline too slow for practical use | Latency benchmarks |

A single end-to-end F1 score would mask which layer is responsible for a regression. The three-layer design localizes failures: if RAGAS context precision drops but F1 stays stable, the retrieval degraded but synthesis compensated. If F1 drops but RAGAS is unchanged, the synthesis agent regressed.

---

## Layer 1 — RAG quality (RAGAS)

### Metrics

RAGAS computes four metrics that measure different aspects of retrieval and synthesis quality:

**Faithfulness** — measures whether every claim in the generated answer is supported by the retrieved context. Computed by: (1) breaking the answer into atomic claims using an LLM, (2) checking each claim against the retrieved chunks, (3) reporting the fraction of claims that are grounded. A faithfulness score of 1.0 means every claim traces to a retrieved source. A score below 0.80 indicates the synthesis agent is generating claims not supported by retrieval — a hallucination signal.

**Answer relevancy** — measures whether the generated answer directly addresses the question asked. Computed by: generating several reverse questions from the answer and measuring cosine similarity between the original question and each reverse question. A high score means the answer stays on topic.

**Context precision** — measures whether the retrieved chunks that are actually useful (relevant to the question) are ranked above irrelevant chunks. A high score means the reranking pipeline is doing its job — useful context is at the top.

**Context recall** — measures whether all the information needed to answer the question is present in the retrieved context. A low score means the retrieval pipeline is missing relevant regulatory clauses.

### Golden dataset construction

The RAGAS golden dataset contains 200 question-answer-context triples, each derived from a specific GDPR article:

```jsonl
{
  "question": "What information must a data controller provide about data retention?",
  "ground_truth": "The data controller must inform the data subject of the period for which personal data will be stored, or if not possible, the criteria used to determine that period, per Article 13(2)(a).",
  "contexts": [
    "GDPR Article 13(2)(a): the period for which the personal data will be stored, or if that is not possible, the criteria used to determine that period..."
  ]
}
```

Construction process:
1. For each of the 99 GDPR articles, write 2–3 compliance questions a real analyst might ask
2. Write the ground truth answer in your own words (not copied from the article)
3. Identify the specific GDPR text that supports the answer
4. For difficult articles, have a second reviewer validate the ground truth

The dataset is committed at `data/golden/gdpr_qa.jsonl` and versioned alongside the codebase. A second dataset for SOC-2 is in progress (`data/golden/soc2_qa.jsonl`, currently 80 examples).

### Running RAGAS evaluation

```bash
python src/eval/run_ragas.py \
  --dataset data/golden/gdpr_qa.jsonl \
  --top-k 5 \
  --scope gdpr \
  --output-mlflow
```

Output:

```
Running RAGAS evaluation on 200 examples...
────────────────────────────────────────────
Faithfulness:        0.912  ✓ (threshold: 0.85)
Answer relevancy:    0.887  ✓ (threshold: 0.82)
Context precision:   0.841  ✓ (threshold: 0.80)
Context recall:      0.863  ✓ (threshold: 0.80)
────────────────────────────────────────────
All RAGAS thresholds passed. Logged to MLflow run: abc123.
```

### Interpreting RAGAS regressions

| Metric drops | Likely cause | Investigation |
|---|---|---|
| Context precision | Reranker degraded; too many irrelevant chunks ranked high | Check reranker model; inspect top-5 chunks for failing examples |
| Context recall | Dense retrieval missing relevant chunks; corpus gap | Check IVFFlat index freshness; run retrieval diagnostic script |
| Faithfulness | Synthesis agent citing claims not in context | Check synthesis prompt; review failing examples in LangSmith |
| Answer relevancy | Synthesis answering a different question | Review synthesis prompt for scope drift |

---

## Layer 2 — Gap detection accuracy (custom F1)

### Why a custom harness rather than RAGAS alone

RAGAS measures whether the RAG pipeline returns good context. It does not measure whether the system correctly identifies that a company policy is missing a required GDPR provision. A pipeline with perfect RAGAS scores could still fail to detect gaps if the synthesis agent reasons incorrectly. The gap detection F1 harness measures the end task.

### Dataset construction

The gap detection dataset contains 150 labeled examples:

```jsonl
{
  "example_id": "gap_001",
  "policy_text": "We retain your personal data for as long as your account is active.",
  "regulation_scope": ["gdpr"],
  "ground_truth_gaps": [
    {
      "has_gap": true,
      "regulatory_article": "GDPR Article 13(2)(a)",
      "gap_type": "retention_period_unspecified"
    }
  ]
}
```

100 positive examples (genuine gaps) are seeded from real SEC enforcement actions available on EDGAR. Each enforcement action cites the exact rule the company violated and includes the company's public filings that failed to comply. This gives ground truth determined by actual regulators, not synthetic annotation.

50 negative examples (no gap) are constructed from policy documents that satisfy the specific regulatory requirement they are paired against — verified by two independent reviewers.

The positive/negative split is intentionally 2:1 to reflect real-world base rates (non-compliant policies are more common than fully compliant ones in the training distribution).

### Evaluation mechanics

For each example, the pipeline runs the full synthesis agent against the policy text and retrieves from the regulatory corpus. The output is a list of detected gaps. We evaluate:

**Precision** = (correct gaps detected) / (total gaps detected by system)  
**Recall** = (correct gaps detected) / (total actual gaps in ground truth)  
**F1** = harmonic mean of precision and recall

A gap is counted as "correct" if:
1. The `regulatory_article` matches the ground truth article (exact string match after normalization)
2. The detected gap's `gap_type` is semantically consistent with the ground truth (evaluated by a secondary LLM judge)

```bash
python src/eval/run_gap_detection.py \
  --dataset data/golden/gap_detection.jsonl \
  --threshold 0.85 \
  --output-mlflow \
  --verbose  # shows per-example breakdown
```

### Threshold sweep

The harness also runs a threshold sweep over the synthesis `confidence` score to produce a precision-recall curve. This is logged to MLflow and helps tune the confidence threshold for production:

```bash
python src/eval/run_gap_detection.py \
  --dataset data/golden/gap_detection.jsonl \
  --sweep-thresholds 0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.95
```

The sweep output helps answer: "at what confidence threshold does precision reach 0.95?" (useful for a high-precision use case like automatic reporting).

### Error analysis

The `--verbose` flag outputs per-example results including failure reasons:

```
Example gap_023: FAIL
  Ground truth: GDPR Article 17(1) (right to erasure)
  Detected:     Nothing (false negative)
  Policy text:  "Users can delete their account via Settings."
  Analysis:     Policy implies erasure capability but doesn't confirm data is
                permanently deleted from backups. Synthesis agent accepted the
                implication; should have flagged the ambiguity.
  Action:       Strengthen synthesis prompt: instruct agent to flag implied
                but unconfirmed compliance as a minor gap.
```

---

## Layer 3 — Agent behavior (LLM-as-judge)

### What this layer evaluates

RAGAS and F1 measure outputs. The LLM-as-judge layer measures agent behavior — the reasoning and tool use decisions that produce those outputs. It catches issues like:

- The router sends a CSV file to the doc agent instead of the data agent (misrouting)
- The synthesis agent fails to use the retrieved chunks and reasons from parametric memory instead (tool use failure)
- The doc agent runs NER on a 3,000-token summary instead of the full document (incorrect tool call)

### How it works

After each job, the LangSmith trace is retrieved via the LangSmith API. A separate evaluator LLM (Claude Sonnet, with a dedicated evaluation system prompt) reads the trace and scores three dimensions:

**Routing correctness (0–5):** Did the `classify_input` node correctly identify the modality of each file and route it to the right agent?

**Tool use quality (0–5):** Did each agent call the right models/tools in the right order? Did it handle edge cases (empty text, unrecognized file type) gracefully?

**Citation accuracy (0–5):** Does each gap in the synthesis output cite a specific regulatory article? Does the cited article exist in the retrieved chunks (i.e., is it grounded in retrieval rather than model memory)?

### Running the agent judge

```bash
# Evaluate traces from the last 24 hours
python src/eval/run_agent_judge.py --lookback-hours 24

# Evaluate traces for a specific job
python src/eval/run_agent_judge.py --job-id 01J3KM9VXKQ4BFHRZS8G7WNPD

# Evaluate a batch of jobs from a file
python src/eval/run_agent_judge.py --job-ids-file data/eval/test_job_ids.txt
```

### Interpreting agent judge scores

Average scores below 3.5 on any dimension warrant investigation:

| Dimension | Score < 3.5 | Common cause |
|---|---|---|
| Routing correctness | < 3.5 | MIME type detection failing; zero-shot classifier underperforming on edge cases |
| Tool use quality | < 3.5 | Agent node error handling silently swallowing exceptions; model calls returning empty results |
| Citation accuracy | < 3.5 | Synthesis agent citing articles not in retrieved chunks (hallucinated citations) |

---

## Nightly CI evaluation

The nightly eval job runs all three layers in sequence and fails the CI build if any threshold is missed:

```yaml
# Thresholds enforced in CI (src/eval/thresholds.py)
RAGAS_FAITHFULNESS_MIN       = 0.85
RAGAS_ANSWER_RELEVANCY_MIN   = 0.82
RAGAS_CONTEXT_PRECISION_MIN  = 0.80
RAGAS_CONTEXT_RECALL_MIN     = 0.80
GAP_DETECTION_F1_MIN         = 0.85
AGENT_ROUTING_ACCURACY_MIN   = 0.95
GROUNDEDNESS_PASS_RATE_MIN   = 0.98
```

When a threshold is missed, the CI job:
1. Fails the build (blocks the PR or flags the nightly run)
2. Opens a GitHub issue with the metric name, current score, threshold, and a link to the MLflow run
3. Posts a summary to the `#meridian-eval` Slack channel (if configured)

### Tracking metric trends in MLflow

Every eval run is logged to MLflow with a full parameter set:

```python
mlflow.log_params({
    "pipeline_version": settings.VERSION,
    "retrieval_top_k": settings.RETRIEVAL_TOP_K_RERANK,
    "reranker_model": settings.RERANKER_MODEL,
    "synthesis_model": settings.SYNTHESIS_MODEL,
    "groundedness_threshold": settings.GROUNDEDNESS_THRESHOLD,
    "dataset_size_ragas": 200,
    "dataset_size_gaps": 150,
})
mlflow.log_metrics({
    "ragas_faithfulness": result.faithfulness,
    "ragas_answer_relevancy": result.answer_relevancy,
    "ragas_context_precision": result.context_precision,
    "ragas_context_recall": result.context_recall,
    "gap_f1": gap_result.f1,
    "gap_precision": gap_result.precision,
    "gap_recall": gap_result.recall,
    "agent_routing_accuracy": judge_result.routing_accuracy,
    "groundedness_pass_rate": groundedness_result.pass_rate,
})
```

Navigate to `http://localhost:5000` (MLflow UI) and open the `meridian-eval` experiment to view metric trend charts over time.

---

## Latency benchmarks

In addition to accuracy metrics, the eval suite measures end-to-end latency:

```bash
python src/eval/run_latency_benchmark.py \
  --scenarios document_only,audio_only,multimodal \
  --samples 20 \
  --output-mlflow
```

Latency is measured as the wall-clock time from job submission to `complete` status, using pre-uploaded fixture files to isolate pipeline latency from upload time.

Enforced CI thresholds:

| Scenario | P50 target | P95 target |
|---|---|---|
| Document only (10-page PDF) | 60s | 120s |
| Audio only (5-minute MP3) | 90s | 150s |
| Multi-modal (PDF + MP3 + CSV) | 100s | 180s |

---

## Calibration and improvement workflow

When an eval metric regresses, the recommended investigation workflow is:

1. **Identify failing examples** with `--verbose` flag
2. **Check LangSmith traces** for those examples — the node-level spans show exactly which step produced the wrong output
3. **Form a hypothesis** — is it the retrieval, the synthesis prompt, or the model?
4. **Make a targeted change** — change one thing at a time
5. **Re-run the affected eval layer** before committing — don't wait for nightly CI
6. **Document the change** in `CHANGELOG.md` with before/after metric scores

This workflow ensures the eval harness is used as a development feedback loop, not just a CI gate.
