"""Model wrappers — HuggingFace Inference API and local models."""
from src.models.registry import MODELS, ModelRegistry
from src.models.nlp import NERModel, QAModel, ZeroShotClassifier, Summarizer
from src.models.retrieval import SimilarityModel, CrossEncoderReranker
from src.models.asr import ASRModel
from src.models.vision import ImageCaptioner, VQAModel
from src.models.tabular import TAPASModel, TimeSeriesForecaster
from src.models.llm import ClaudeClient

__all__ = [
    "MODELS", "ModelRegistry",
    "NERModel", "QAModel", "ZeroShotClassifier", "Summarizer",
    "SimilarityModel", "CrossEncoderReranker",
    "ASRModel", "ImageCaptioner", "VQAModel",
    "TAPASModel", "TimeSeriesForecaster",
    "ClaudeClient",
]
