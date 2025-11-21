from PIL import Image

from zocr.ocr_pipeline import mocks
from zocr.ocr_pipeline import cli


def test_build_pipeline_with_mocks():
    pipeline = cli.build_document_pipeline(use_mocks=True)
    assert isinstance(pipeline.input_handler, mocks.MockInputHandler)
    assert isinstance(pipeline.page_pipeline.segmenter, mocks.MockSegmenter)
    assert isinstance(pipeline.page_pipeline.region_classifier, mocks.MockRegionClassifier)
    assert isinstance(pipeline.page_pipeline.text_ocr, mocks.MockTextOCR)
    assert isinstance(pipeline.page_pipeline.vllm, mocks.MockVLLM)
    assert isinstance(pipeline.page_pipeline.table_extractor, mocks.MockTableExtractor)
    assert isinstance(pipeline.page_pipeline.aggregator, mocks.MockAggregator)


def test_cli_runs_with_mock_components(tmp_path):
    img_path = tmp_path / "page.png"
    Image.new("RGB", (10, 10), color="white").save(img_path)

    out_path = tmp_path / "out.json"
    cli.main([
        "--images",
        img_path.as_posix(),
        "--out",
        out_path.as_posix(),
        "--use-mocks",
        "--document-id",
        "doc-123",
    ])

    payload = out_path.read_text(encoding="utf-8")
    assert "doc-123" in payload
    assert "text-1" in payload
