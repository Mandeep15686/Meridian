"""Model wrappers — HuggingFace Inference API and local models."""

from src.models.asr import ASRModel
from src.models.llm import ClaudeClient
from src.models.nlp import NERModel, QAModel, Summarizer, ZeroShotClassifier
from src.models.registry import MODELS, ModelRegistry
from src.models.retrieval import CrossEncoderReranker, SimilarityModel
from src.models.tabular import TAPASModel, TimeSeriesForecaster
from src.models.vision import ImageCaptioner, VQAModel

__all__ = [
    "MODELS", "ModelRegistry",
    "NERModel", "QAModel", "ZeroShotClassifier", "Summarizer",
    "SimilarityModel", "CrossEncoderReranker",
    "ASRModel", "ImageCaptioner", "VQAModel",
    "TAPASModel", "TimeSeriesForecaster",
    "ClaudeClient",
]
