"""Minimal built-in implementations for the OCR pipeline.

These classes are intentionally lightweight so the pipeline can run without
external ML models while still exercising the end-to-end flow on real images.
"""
from __future__ import annotations

from typing import List

from PIL import Image

from .interfaces import Aggregator, RegionClassifier, Segmenter, TableExtractor, VLLM
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


class FullPageSegmenter(Segmenter):
    """Return a single region that spans the entire image.

    This is a pragmatic baseline useful for simple documents or debugging
    the pipeline wiring. The original page image is preserved as the region
    crop so downstream OCR components can operate without additional access
    to the source page.
    """

    def __init__(self, confidence: float = 1.0) -> None:
        self.confidence = confidence

    def segment(self, page: PageInput) -> List[SegmentedRegion]:
        image = page.image
        if not isinstance(image, Image.Image):
            raise TypeError("FullPageSegmenter expects a PIL.Image instance")

        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError("Page image must have positive dimensions")

        bounding_box = BoundingBox(x=0, y=0, width=width, height=height)
        region_id = f"{page.document_id}-page{page.page_number}-full"
        return [
            SegmentedRegion(
                region_id=region_id,
                bounding_box=bounding_box,
                image_crop=image,
                confidence=self.confidence,
                reading_order=0,
            )
        ]


class AspectRatioRegionClassifier(RegionClassifier):
    """Classify regions using only bounding-box geometry.

    * Very wide regions are treated as tables
    * Near-square regions are treated as images
    * Everything else defaults to text
    """

    def __init__(
        self,
        table_aspect_ratio: float = 2.5,
        square_tolerance: float = 0.15,
        default_type: RegionType = RegionType.TEXT,
    ) -> None:
        self.table_aspect_ratio = table_aspect_ratio
        self.square_tolerance = square_tolerance
        self.default_type = default_type

    def classify(self, region: SegmentedRegion) -> ClassifiedRegion:
        width = region.bounding_box.width
        height = region.bounding_box.height
        ratio = width / height if height else 0

        if ratio >= self.table_aspect_ratio:
            classification = RegionType.TABLE
        elif abs(1 - ratio) <= self.square_tolerance:
            classification = RegionType.IMAGE
        else:
            classification = self.default_type

        return ClassifiedRegion(
            region_id=region.region_id,
            bounding_box=region.bounding_box,
            classification=classification,
            confidence=region.confidence,
            reading_order=region.reading_order,
            image_crop=region.image_crop,
        )


class SimpleAggregator(Aggregator):
    """Aggregate component outputs into a document payload."""

    def aggregate(
        self,
        page: PageInput,
        classified_regions: List[ClassifiedRegion],
        text_results: List[TextOcrResult],
        image_results: List[ImageCaptionResult],
        table_results: List[TableExtractionResult],
    ) -> DocumentOutput:
        text_map = {item.region_id: item for item in text_results}
        image_map = {item.region_id: item for item in image_results}
        table_map = {item.region_id: item for item in table_results}

        sorted_regions = sorted(
            classified_regions,
            key=lambda r: (
                r.reading_order if r.reading_order is not None else float("inf"),
                r.region_id,
            ),
        )

        region_outputs: List[RegionOutput] = []
        for region in sorted_regions:
            if region.classification == RegionType.TEXT and region.region_id in text_map:
                content = {
                    "text": text_map[region.region_id].text,
                    "confidence": text_map[region.region_id].confidence,
                    "language": text_map[region.region_id].language,
                }
            elif region.classification == RegionType.IMAGE and region.region_id in image_map:
                content = {
                    "caption": image_map[region.region_id].caption,
                    "confidence": image_map[region.region_id].confidence,
                    "detailed_description": image_map[region.region_id].detailed_description,
                    "detected_objects": image_map[region.region_id].detected_objects,
                }
            elif region.classification == RegionType.TABLE and region.region_id in table_map:
                content = {
                    "table_data": table_map[region.region_id].table_data.model_dump(),
                    "confidence": table_map[region.region_id].confidence,
                    "format": table_map[region.region_id].format,
                }
            else:
                content = {}

            region_outputs.append(
                RegionOutput(
                    region_id=region.region_id,
                    type=region.classification,
                    bounding_box=region.bounding_box,
                    reading_order=region.reading_order,
                    content=content,
                )
            )

        metadata = DocumentMetadata(
            total_regions=len(region_outputs),
            text_regions=sum(1 for region in region_outputs if region.type == RegionType.TEXT),
            image_regions=sum(1 for region in region_outputs if region.type == RegionType.IMAGE),
            table_regions=sum(1 for region in region_outputs if region.type == RegionType.TABLE),
        )

        return DocumentOutput(
            document_id=page.document_id,
            page_number=page.page_number,
            regions=region_outputs,
            metadata=metadata,
        )


class DummyVLLM(VLLM):
    """Lightweight captioner for non-text regions when a real VLM is unavailable."""

    def __init__(self, confidence: float = 0.5) -> None:
        self.confidence = confidence

    def describe(self, region: ClassifiedRegion) -> ImageCaptionResult:
        return ImageCaptionResult(
            region_id=region.region_id,
            caption=f"Image region {region.region_id}",
            confidence=self.confidence,
            detected_objects=None,
            detailed_description=None,
        )


class DummyTableExtractor(TableExtractor):
    """Return placeholder table data for structured regions."""

    def __init__(self, confidence: float = 0.5) -> None:
        self.confidence = confidence

    def extract(self, region: ClassifiedRegion) -> TableExtractionResult:
        table_data = TableData(
            headers=["col1", "col2"],
            rows=[{"col1": "value1", "col2": "value2"}],
            num_rows=1,
            num_columns=2,
        )
        return TableExtractionResult(
            region_id=region.region_id,
            table_data=table_data,
            confidence=self.confidence,
            format="simple",
        )

