"""Mock implementations of OCR pipeline components for testing."""
from __future__ import annotations

from typing import Dict, List

from .interfaces import Aggregator, InputHandler, RegionClassifier, Segmenter, TableExtractor, TextOCR, VLLM
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


class MockSegmenter(Segmenter):
    def __init__(self) -> None:
        self.calls: List[PageInput] = []

    def segment(self, page: PageInput) -> List[SegmentedRegion]:
        self.calls.append(page)
        base_box = BoundingBox(x=0, y=0, width=100, height=100)
        return [
            SegmentedRegion(region_id="text-1", bounding_box=base_box, image_crop=None, confidence=0.9, reading_order=0),
            SegmentedRegion(region_id="image-1", bounding_box=base_box, image_crop=None, confidence=0.92, reading_order=1),
            SegmentedRegion(region_id="table-1", bounding_box=base_box, image_crop=None, confidence=0.93, reading_order=2),
        ]


class MockRegionClassifier(RegionClassifier):
    def __init__(self) -> None:
        self.calls: List[SegmentedRegion] = []

    def classify(self, region: SegmentedRegion) -> ClassifiedRegion:
        self.calls.append(region)
        if region.region_id.startswith("text"):
            classification = RegionType.TEXT
        elif region.region_id.startswith("image"):
            classification = RegionType.IMAGE
        else:
            classification = RegionType.TABLE
        return ClassifiedRegion(
            region_id=region.region_id,
            bounding_box=region.bounding_box,
            classification=classification,
            confidence=region.confidence,
            reading_order=region.reading_order,
            image_crop=region.image_crop,
        )


class MockInputHandler(InputHandler):
    def __init__(self) -> None:
        self.calls: List[DocumentInput] = []

    def load(self, document: DocumentInput) -> List[PageInput]:
        self.calls.append(document)
        images = document.images or []
        if not images:
            raise ValueError("MockInputHandler requires images in DocumentInput")

        return [
            PageInput(
                document_id=document.document_id,
                page_number=idx + 1,
                image=image,
                dpi=document.dpi,
            )
            for idx, image in enumerate(images)
        ]


class MockTextOCR(TextOCR):
    def __init__(self) -> None:
        self.calls: List[ClassifiedRegion] = []

    def run(self, region: ClassifiedRegion) -> TextOcrResult:
        self.calls.append(region)
        return TextOcrResult(
            region_id=region.region_id,
            text=f"text content for {region.region_id}",
            confidence=0.95,
            language="en",
        )


class MockVLLM(VLLM):
    def __init__(self) -> None:
        self.calls: List[ClassifiedRegion] = []

    def describe(self, region: ClassifiedRegion) -> ImageCaptionResult:
        self.calls.append(region)
        return ImageCaptionResult(
            region_id=region.region_id,
            caption=f"caption for {region.region_id}",
            confidence=0.9,
        )


class MockTableExtractor(TableExtractor):
    def __init__(self) -> None:
        self.calls: List[ClassifiedRegion] = []

    def extract(self, region: ClassifiedRegion) -> TableExtractionResult:
        self.calls.append(region)
        table_data = TableData(headers=["col1", "col2"], rows=[{"col1": "a", "col2": "b"}], num_rows=1, num_columns=2)
        return TableExtractionResult(region_id=region.region_id, table_data=table_data, confidence=0.88)


class MockAggregator(Aggregator):
    def __init__(self) -> None:
        self.calls: List[Dict[str, object]] = []

    def aggregate(
        self,
        page: PageInput,
        classified_regions: List[ClassifiedRegion],
        text_results: List[TextOcrResult],
        image_results: List[ImageCaptionResult],
        table_results: List[TableExtractionResult],
    ) -> DocumentOutput:
        payload = {
            "page": page,
            "classified_regions": classified_regions,
            "text_results": text_results,
            "image_results": image_results,
            "table_results": table_results,
        }
        self.calls.append(payload)

        text_map: Dict[str, TextOcrResult] = {item.region_id: item for item in text_results}
        image_map: Dict[str, ImageCaptionResult] = {item.region_id: item for item in image_results}
        table_map: Dict[str, TableExtractionResult] = {item.region_id: item for item in table_results}

        sorted_regions = sorted(
            classified_regions,
            key=lambda r: (r.reading_order if r.reading_order is not None else float("inf"), r.region_id),
        )

        region_outputs = []
        for region in sorted_regions:
            if region.classification == RegionType.TEXT and region.region_id in text_map:
                content = {
                    "text": text_map[region.region_id].text,
                    "confidence": text_map[region.region_id].confidence,
                }
            elif region.classification == RegionType.IMAGE and region.region_id in image_map:
                content = {
                    "caption": image_map[region.region_id].caption,
                    "confidence": image_map[region.region_id].confidence,
                }
            elif region.classification == RegionType.TABLE and region.region_id in table_map:
                content = {
                    "table_data": table_map[region.region_id].table_data.model_dump(),
                    "confidence": table_map[region.region_id].confidence,
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
            text_regions=sum(1 for r in region_outputs if r.type == RegionType.TEXT),
            image_regions=sum(1 for r in region_outputs if r.type == RegionType.IMAGE),
            table_regions=sum(1 for r in region_outputs if r.type == RegionType.TABLE),
        )

        return DocumentOutput(
            document_id=page.document_id,
            page_number=page.page_number,
            regions=region_outputs,
            metadata=metadata,
        )

