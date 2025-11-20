"""Interfaces for OCR pipeline components."""
from __future__ import annotations

from typing import List, Protocol

from .models import (
    ClassifiedRegion,
    DocumentOutput,
    DocumentInput,
    ImageCaptionResult,
    PageInput,
    SegmentedRegion,
    TableExtractionResult,
    TextOcrResult,
)


class InputHandler(Protocol):
    def load(self, document: DocumentInput) -> List[PageInput]:
        ...


class Segmenter(Protocol):
    def segment(self, page: PageInput) -> List[SegmentedRegion]:
        ...


class RegionClassifier(Protocol):
    def classify(self, region: SegmentedRegion) -> ClassifiedRegion:
        ...


class TextOCR(Protocol):
    def run(self, region: ClassifiedRegion) -> TextOcrResult:
        ...


class VLLM(Protocol):
    def describe(self, region: ClassifiedRegion) -> ImageCaptionResult:
        ...


class TableExtractor(Protocol):
    def extract(self, region: ClassifiedRegion) -> TableExtractionResult:
        ...


class Aggregator(Protocol):
    def aggregate(
        self,
        page: PageInput,
        classified_regions: List[ClassifiedRegion],
        text_results: List[TextOcrResult],
        image_results: List[ImageCaptionResult],
        table_results: List[TableExtractionResult],
    ) -> DocumentOutput:
        ...


class OcrPipeline(Protocol):
    def process(self, page: PageInput) -> DocumentOutput:
        ...

