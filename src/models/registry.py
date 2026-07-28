"""Model registry — all HuggingFace model IDs as typed constants."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelRegistry:
    # NLP
    qa: str = "deepset/roberta-base-squad2"
    ner: str = "dslim/bert-base-NER"
    classifier: str = "facebook/bart-large-mnli"
    summarizer: str = "facebook/bart-large-cnn"

    # Retrieval
    reranker: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    similarity: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedder: str = "text-embedding-3-small"  # OpenAI via LlamaIndex

    # Audio
    asr: str = "openai/whisper-large-v3"
    diarizer: str = "pyannote/speaker-diarization-3.1"

    # Vision
    captioner: str = "Salesforce/blip2-opt-2.7b"
    vqa: str = "dandelin/vilt-b32-finetuned-vqa"
    colpali: str = "vidore/colpali-v1.2"

    # Tabular
    tapas: str = "google/tapas-base-finetuned-wtq"
    forecaster: str = "amazon/chronos-t5-small"

    # LLM
    llm: str = "claude-sonnet-4-6"


MODELS = ModelRegistry()
