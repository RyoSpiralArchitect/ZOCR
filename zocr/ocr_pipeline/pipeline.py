# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024 ZOCR contributors

"""Composable OCR pipeline implementation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .interfaces import (
    Aggregator,
    DocumentOcrPipeline,
    InputHandler,
    OcrPipeline as OcrPipelineProtocol,
    RegionClassifier,
    Segmenter,
    TableExtractor,
    TextOCR,
    VLLM,
)
from .models import ClassifiedRegion, DocumentInput, DocumentOutput, PageInput, RegionType, SegmentedRegion


@dataclass
class OcrPipeline(OcrPipelineProtocol):
    segmenter: Segmenter
    region_classifier: RegionClassifier
    text_ocr: TextOCR
    vllm: VLLM
    table_extractor: TableExtractor
    aggregator: Aggregator

    def process(self, page: PageInput) -> DocumentOutput:
        segments: List[SegmentedRegion] = self.segmenter.segment(page)
        classified: List[ClassifiedRegion] = [
            self.region_classifier.classify(region) for region in segments
        ]

        text_results = []
        image_results = []
        table_results = []

        for region in classified:
            if region.classification == RegionType.TEXT:
                text_results.append(self.text_ocr.run(region))
            elif region.classification == RegionType.IMAGE:
                image_results.append(self.vllm.describe(region))
            elif region.classification == RegionType.TABLE:
                table_results.append(self.table_extractor.extract(region))

        return self.aggregator.aggregate(
            page=page,
            classified_regions=classified,
            text_results=text_results,
            image_results=image_results,
            table_results=table_results,
        )


@dataclass
class DocumentPipeline(DocumentOcrPipeline):
    input_handler: InputHandler
    page_pipeline: OcrPipeline

    def process(self, document: DocumentInput) -> List[DocumentOutput]:
        pages = self.input_handler.load(document)
        return [self.page_pipeline.process(page) for page in pages]

