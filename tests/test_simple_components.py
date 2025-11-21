# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024 ZOCR contributors

from PIL import Image

from zocr.ocr_pipeline import (
    AspectRatioRegionClassifier,
    BoundingBox,
    DummyTableExtractor,
    DummyVLLM,
    FullPageSegmenter,
    MockTextOCR,
    OcrPipeline,
    PageInput,
    RegionType,
    SegmentedRegion,
    SimpleAggregator,
)


def test_full_page_segmenter_returns_single_region():
    page = PageInput(document_id="doc-1", page_number=1, image=Image.new("RGB", (100, 50)))

    segmenter = FullPageSegmenter()
    regions = segmenter.segment(page)

    assert len(regions) == 1
    region = regions[0]
    assert region.bounding_box.width == 100
    assert region.bounding_box.height == 50
    assert region.image_crop.size == (100, 50)


def test_aspect_ratio_region_classifier_uses_geometry():
    classifier = AspectRatioRegionClassifier(table_aspect_ratio=2.0, square_tolerance=0.1)

    wide_region = SegmentedRegion(
        region_id="r1",
        bounding_box=BoundingBox(x=0, y=0, width=200, height=50),
        image_crop=None,
        confidence=0.9,
        reading_order=0,
    )
    square_region = SegmentedRegion(
        region_id="r2",
        bounding_box=BoundingBox(x=0, y=0, width=100, height=95),
        image_crop=None,
        confidence=0.9,
        reading_order=1,
    )
    tall_region = SegmentedRegion(
        region_id="r3",
        bounding_box=BoundingBox(x=0, y=0, width=80, height=200),
        image_crop=None,
        confidence=0.9,
        reading_order=2,
    )

    assert classifier.classify(wide_region).classification == RegionType.TABLE
    assert classifier.classify(square_region).classification == RegionType.IMAGE
    assert classifier.classify(tall_region).classification == RegionType.TEXT


def test_simple_pipeline_with_basic_components():
    image = Image.new("RGB", (120, 80))
    page = PageInput(document_id="doc-2", page_number=1, image=image)

    segmenter = FullPageSegmenter()
    classifier = AspectRatioRegionClassifier(table_aspect_ratio=10.0)
    pipeline = OcrPipeline(
        segmenter=segmenter,
        region_classifier=classifier,
        text_ocr=MockTextOCR(),
        vllm=DummyVLLM(),
        table_extractor=DummyTableExtractor(),
        aggregator=SimpleAggregator(),
    )

    output = pipeline.process(page)

    assert output.document_id == "doc-2"
    assert output.page_number == 1
    assert output.metadata.total_regions == 1
    assert output.metadata.text_regions == 1
    assert output.regions[0].content["text"].startswith("text content")

