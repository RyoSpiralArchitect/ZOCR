"""Modular OCR pipeline scaffolding."""

from .input_handler import BasicInputHandler
from .interfaces import (
    Aggregator,
    InputHandler,
    OcrPipeline,
    RegionClassifier,
    Segmenter,
    TableExtractor,
    TextOCR,
    VLLM,
)
from .mocks import MockAggregator, MockInputHandler, MockRegionClassifier, MockSegmenter, MockTableExtractor, MockTextOCR, MockVLLM
from .models import (
    BoundingBox,
    ClassifiedRegion,
    DocumentInput,
    DocumentMetadata,
    DocumentOutput,
    ImageCaptionResult,
    PageInput,
    RegionOutput,
    RegionType,
    SegmentedRegion,
    TableData,
    TableExtractionResult,
    TextOcrResult,
)
from .pipeline import OcrPipeline
from .tesseract import TesseractTextOCR

__all__ = [
    "Aggregator",
    "BasicInputHandler",
    "BoundingBox",
    "ClassifiedRegion",
    "DocumentInput",
    "DocumentMetadata",
    "DocumentOutput",
    "ImageCaptionResult",
    "InputHandler",
    "MockAggregator",
    "MockInputHandler",
    "MockRegionClassifier",
    "MockSegmenter",
    "MockTableExtractor",
    "MockTextOCR",
    "MockVLLM",
    "OcrPipeline",
    "PageInput",
    "RegionClassifier",
    "RegionOutput",
    "RegionType",
    "SegmentedRegion",
    "Segmenter",
    "TableData",
    "TableExtractionResult",
    "TableExtractor",
    "TesseractTextOCR",
    "TextOCR",
    "TextOcrResult",
    "VLLM",
]

