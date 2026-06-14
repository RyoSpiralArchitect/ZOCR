from PIL import Image, ImageDraw

from zocr.ocr_pipeline import (
    AspectRatioRegionClassifier,
    BoundingBox,
    ClassifiedRegion,
    DummyTableExtractor,
    DummyVLLM,
    FullPageSegmenter,
    MockTextOCR,
    OcrPipeline,
    PageInput,
    RegionType,
    SegmentedRegion,
    SimpleAggregator,
    SimpleTableExtractor,
    SimpleVLLM,
)
from zocr.ocr_pipeline import simple as simple_components


def test_full_page_segmenter_returns_single_region():
    page = PageInput(document_id="doc-1", page_number=1, image=Image.new("RGB", (100, 50)))

    segmenter = FullPageSegmenter()
    regions = segmenter.segment(page)

    assert len(regions) == 1
    region = regions[0]
    assert region.bounding_box.width == 100
    assert region.bounding_box.height == 50
    assert region.image_crop.size == (100, 50)


def test_full_page_segmenter_splits_two_column_layout():
    image = Image.new("RGB", (220, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 75, 100), fill="black")
    draw.rectangle((145, 20, 200, 100), fill="black")
    page = PageInput(document_id="doc-columns", page_number=1, image=image)

    segmenter = FullPageSegmenter(min_region_fraction=0.04, min_gap_fraction=0.03)
    regions = segmenter.segment(page)

    assert len(regions) == 2
    assert regions[0].bounding_box.x < regions[1].bounding_box.x
    assert regions[0].bounding_box.width < 90
    assert regions[1].bounding_box.width < 90


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


def test_simple_vllm_uses_provider_result_when_available():
    region = ClassifiedRegion(
        region_id="img-1",
        bounding_box=BoundingBox(x=0, y=0, width=40, height=30),
        classification=RegionType.IMAGE,
        confidence=0.9,
        reading_order=0,
        image_crop=Image.new("RGB", (40, 30), "white"),
    )

    vllm = SimpleVLLM(
        captioner=lambda _region: {
            "caption": "wiring diagram with labeled terminals",
            "confidence": 0.88,
            "detected_objects": ["terminal"],
        }
    )

    result = vllm.describe(region)

    assert result.caption == "wiring diagram with labeled terminals"
    assert result.confidence == 0.88
    assert result.detected_objects == ["terminal"]


def test_table_extractor_clusters_rows_and_columns(monkeypatch):
    class FakeOutput:
        DICT = object()

    class FakeTesseract:
        Output = FakeOutput

        @staticmethod
        def image_to_data(_image, output_type):  # noqa: ANN001
            assert output_type is FakeOutput.DICT
            return {
                "text": ["Item", "Qty", "Price", "Bolt", "2", "$10"],
                "conf": ["93", "91", "92", "89", "95", "90"],
                "left": [10, 90, 150, 10, 90, 150],
                "top": [10, 10, 10, 35, 35, 35],
                "width": [34, 28, 42, 34, 12, 28],
                "height": [10, 10, 10, 10, 10, 10],
                "line_num": [1, 1, 1, 2, 2, 2],
            }

    monkeypatch.setattr(simple_components, "pytesseract", FakeTesseract)
    region = ClassifiedRegion(
        region_id="table-1",
        bounding_box=BoundingBox(x=0, y=0, width=220, height=80),
        classification=RegionType.TABLE,
        confidence=0.9,
        reading_order=0,
        image_crop=Image.new("RGB", (220, 80), "white"),
    )

    result = SimpleTableExtractor().extract(region)

    assert result.format == "tesseract_grid"
    assert result.table_data.headers == ["Item", "Qty", "Price"]
    assert result.table_data.rows == [{"Item": "Bolt", "Qty": "2", "Price": "$10"}]
    assert result.confidence >= 0.8


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
