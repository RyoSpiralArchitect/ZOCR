import json

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


def test_cli_runs_input_dir_with_mock_components(tmp_path):
    input_dir = tmp_path / "pages"
    input_dir.mkdir()
    Image.new("RGB", (10, 10), color="white").save(input_dir / "page-2.png")
    Image.new("RGB", (10, 10), color="white").save(input_dir / "page-1.png")

    out_path = tmp_path / "out.json"
    cli.main([
        "--input-dir",
        input_dir.as_posix(),
        "--out",
        out_path.as_posix(),
        "--use-mocks",
        "--document-id",
        "dir-doc",
    ])

    payload = out_path.read_text(encoding="utf-8")
    data = json.loads(payload)
    assert "dir-doc" in payload
    assert len(data) == 2
    assert all(page["document_id"] == "dir-doc" for page in data)


def test_cli_runs_batch_dir_with_mock_components(tmp_path):
    batch_dir = tmp_path / "batch"
    for name in ("doc-a", "doc-b"):
        doc_dir = batch_dir / name
        doc_dir.mkdir(parents=True)
        Image.new("RGB", (10, 10), color="white").save(doc_dir / "page.png")

    out_path = tmp_path / "batch.json"
    cli.main([
        "--batch-dir",
        batch_dir.as_posix(),
        "--out",
        out_path.as_posix(),
        "--use-mocks",
    ])

    payload = out_path.read_text(encoding="utf-8")
    assert "doc-a" in payload
    assert "doc-b" in payload


def test_cli_writes_handoff_run_folder_with_mock_components(tmp_path, capsys):
    img_path = tmp_path / "page.png"
    Image.new("RGB", (10, 10), color="white").save(img_path)
    outdir = tmp_path / "run"

    cli.main([
        "--images",
        img_path.as_posix(),
        "--outdir",
        outdir.as_posix(),
        "--use-mocks",
        "--document-id",
        "handoff-doc",
    ])

    captured = capsys.readouterr()
    assert "ZOCR simple run complete" in captured.out
    assert "manifest_json" in captured.out

    pages = json.loads((outdir / "pages.json").read_text(encoding="utf-8"))
    summary = json.loads((outdir / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((outdir / "manifest.json").read_text(encoding="utf-8"))
    regions = [
        json.loads(line)
        for line in (outdir / "regions.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert pages[0]["document_id"] == "handoff-doc"
    assert summary["schema"] == "zocr.simple_run.v1"
    assert summary["totals"]["documents"] == 1
    assert summary["totals"]["pages"] == 1
    assert summary["totals"]["regions"] == 3
    assert manifest["artifacts"]["pages_json"].endswith("pages.json")
    assert {row["type"] for row in regions} == {"text", "image", "table"}


def test_cli_outdir_and_stdout_json_keep_summary_on_stderr(tmp_path, capsys):
    img_path = tmp_path / "page.png"
    Image.new("RGB", (10, 10), color="white").save(img_path)
    outdir = tmp_path / "run"

    cli.main([
        "--images",
        img_path.as_posix(),
        "--out",
        "-",
        "--outdir",
        outdir.as_posix(),
        "--use-mocks",
        "--document-id",
        "stdout-doc",
    ])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload[0]["document_id"] == "stdout-doc"
    assert "ZOCR simple run complete" in captured.err


def test_cli_rejects_multiple_input_modes(tmp_path):
    img_path = tmp_path / "page.png"
    Image.new("RGB", (10, 10), color="white").save(img_path)
    input_dir = tmp_path / "pages"
    input_dir.mkdir()

    try:
        cli.main([
            "--images",
            img_path.as_posix(),
            "--input-dir",
            input_dir.as_posix(),
            "--use-mocks",
        ])
    except SystemExit as exc:
        assert "exactly one" in str(exc)
    else:
        raise AssertionError("expected SystemExit")
