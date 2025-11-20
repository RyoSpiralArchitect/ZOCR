"""Modular OCR pipeline scaffolding."""

from .interfaces import Aggregator, OcrPipeline, RegionClassifier, Segmenter, TableExtractor, TextOCR, VLLM
from .mocks import MockAggregator, MockRegionClassifier, MockSegmenter, MockTableExtractor, MockTextOCR, MockVLLM
from .models import (
    BoundingBox,
    ClassifiedRegion,
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

__all__ = [
    "Aggregator",
    "BoundingBox",
    "ClassifiedRegion",
    "DocumentMetadata",
    "DocumentOutput",
    "ImageCaptionResult",
    "MockAggregator",
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
    "TextOCR",
    "TextOcrResult",
    "VLLM",
]

