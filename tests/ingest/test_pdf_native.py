import hashlib
import io
import json

import pytest
from PIL import Image

fitz = pytest.importorskip("fitz")

from zocr.ingest.native_pdf import extract_native_page


def _make_pdf(path, *, text="Native evidence 14", hidden=False, raster=False, rotation=0):
    with fitz.open() as doc:
        page = doc.new_page(width=300, height=200)
        if text:
            page.insert_text((30, 50), text, render_mode=3 if hidden else 0)
        if raster:
            buffer = io.BytesIO()
            Image.new("RGB", (30, 30), "gray").save(buffer, format="PNG")
            page.insert_image(fitz.Rect(180, 80, 240, 140), stream=buffer.getvalue())
        page.set_rotation(rotation)
        doc.save(path)


def _render(path):
    with fitz.open(path) as doc:
        pix = doc[0].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def test_native_text_preserves_provenance_without_inventing_ocr_confidence(tmp_path):
    path = tmp_path / "native.pdf"
    _make_pdf(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    result = extract_native_page(path, 0, _render(path), tmp_path / "out", page_number=4)
    assert result["route"] == "native"
    assert result["ocr_regions"] == []
    record, = result["records"]
    assert record["text"] == "Native evidence 14"
    assert record["page"] == 4
    assert record["table_index"] is None
    assert record["row"] is None and record["col"] is None
    assert record["meta"]["source_kind"] == "pdf_text_layer"
    assert record["meta"]["source_sha256"] == digest
    assert record["meta"]["source_page"] == 1
    assert record["meta"]["observed_text"] == record["text"]
    assert record["meta"]["confidence"] == {}
    assert record["meta"]["needs_review"] is True
    assert (tmp_path / "out" / record["image_path"]).is_file()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_image_only_and_hidden_text_require_raster_ocr(tmp_path):
    for name, options, reason in (("scan", {"text": "", "raster": True}, "no_usable_text"),
                                  ("hidden", {"hidden": True}, "hidden_text_layer")):
        path = tmp_path / f"{name}.pdf"
        _make_pdf(path, **options)
        image = _render(path)
        result = extract_native_page(path, 0, image, tmp_path / name)
        assert result["route"] == "raster"
        assert result["reason"] == reason
        assert result["records"] == []
        assert result["ocr_regions"] == [[0, 0, image.width, image.height]]


def test_mixed_page_keeps_embedded_images_as_unread_ocr_regions(tmp_path):
    path = tmp_path / "mixed.pdf"
    _make_pdf(path, raster=True)
    result = extract_native_page(path, 0, _render(path), tmp_path / "out")
    assert result["route"] == "mixed"
    assert result["records"]
    assert result["ocr_regions"] == [[360, 160, 480, 280]]


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_rotated_page_bboxes_map_to_the_archived_pixels(tmp_path, rotation):
    path = tmp_path / "rotated.pdf"
    _make_pdf(path, rotation=rotation)
    image = _render(path)
    result = extract_native_page(path, 0, image, tmp_path / "out")
    record, = result["records"]
    x0, y0, x1, y1 = record["bbox"]
    assert 0 <= x0 < x1 <= image.width
    assert 0 <= y0 < y1 <= image.height
    crop = image.crop((x0, y0, x1, y1)).convert("L")
    assert crop.getextrema()[0] < 100
    assert record["meta"]["page_rotation"] == rotation


def test_native_export_bypasses_image_ocr_and_retains_non_tabular_identity(tmp_path, monkeypatch):
    from zocr.consensus import toy_runtime

    source = tmp_path / "native.pdf"
    _make_pdf(source)
    image = _render(source)
    image_path = tmp_path / "page.png"
    image.save(image_path)
    out = tmp_path / "out"
    plan = extract_native_page(source, 0, image, out)
    doc = tmp_path / "doc.json"
    doc.write_text(json.dumps({"doc_id": "doc", "pages": [{"index": 0, "tables": []}]}))

    def forbidden(*args, **kwargs):
        raise AssertionError("Native PDF text must not pass through image OCR")

    monkeypatch.setattr(toy_runtime, "_resolve_ocr_backend", forbidden)
    target = out / "doc.contextual.jsonl"
    assert toy_runtime.export_jsonl_with_ocr(str(doc), {0: str(image_path)}, str(target), native_pages={0: plan}) == 1
    record = json.loads(target.read_text())
    assert record["text"] == "Native evidence 14"
    assert record["table_index"] is None
    assert record["meta"]["confidence"] == {}
    assert record["meta"]["needs_review"]
    assert toy_runtime.last_export_stats()["native_pdf"]["records"] == 1


def test_native_export_honors_the_record_budget(tmp_path, monkeypatch):
    from zocr.consensus import toy_runtime

    source = tmp_path / "native.pdf"
    _make_pdf(source, text="one\ntwo\nthree")
    image = _render(source)
    image_path = tmp_path / "page.png"
    image.save(image_path)
    out = tmp_path / "out"
    plan = extract_native_page(source, 0, image, out)
    doc = tmp_path / "doc.json"
    doc.write_text(json.dumps({"doc_id": "doc", "pages": [{"index": 0, "tables": []}]}))
    monkeypatch.setenv("ZOCR_EXPORT_MAX_CELLS", "2")
    target = out / "doc.contextual.jsonl"
    assert toy_runtime.export_jsonl_with_ocr(str(doc), {0: str(image_path)}, str(target), native_pages={0: plan}) == 2
    assert len(target.read_text().splitlines()) == 2
    assert toy_runtime.last_export_stats()["truncated"]


def test_real_pipeline_prefers_native_pdf_text_and_keeps_review_required(tmp_path):
    from zocr.service.executor import execute_pipeline
    from zocr.validate import validate_path

    source = tmp_path / "native.pdf"
    _make_pdf(source)
    out = tmp_path / "out"
    summary = execute_pipeline(timeout_sec=60, inputs=[str(source)], outdir=str(out),
                               do_tune=False, toy_lite=True, sweeps_fixed=1, max_cells=32, k=1)
    records = [json.loads(line) for line in (out / "rag/cells.jsonl").read_text().splitlines()]
    assert any(record["text"] == "Native evidence 14" for record in records)
    assert all(record["meta"]["source_kind"] == "pdf_text_layer" for record in records)
    assert summary["pdf_native"]["0"]["route"] == "native"
    assert summary["publication_status"] == "review_required"
    assert summary["quality_status"] == "warn"
    assert validate_path(out, strict=True, max_jsonl_lines=1000, write_manifest_flag=False) == 0
    from zocr.diff.differ import SemanticDiffer
    assert SemanticDiffer().compare_bundle(out / "rag/cells.jsonl", out / "rag/cells.jsonl")["events"] == []


def test_native_export_separates_pdfs_with_repeated_source_page_numbers(tmp_path):
    from zocr.consensus import toy_runtime

    out = tmp_path / "out"
    plans, pages, images = {}, [], {}
    for index in range(2):
        source = tmp_path / f"source-{index}.pdf"
        _make_pdf(source, text=f"Source {index}")
        image = _render(source)
        image_path = tmp_path / f"page-{index}.png"
        image.save(image_path)
        plans[index] = extract_native_page(source, 0, image, out, page_number=index)
        images[index] = str(image_path)
        pages.append({"index": 1, "image_path": str(image_path), "tables": []})
    doc = tmp_path / "doc.json"
    doc.write_text(json.dumps({"doc_id": "doc", "pages": pages}))
    target = out / "doc.contextual.jsonl"
    assert toy_runtime.export_jsonl_with_ocr(str(doc), images, str(target), native_pages=plans) == 2
    records = [json.loads(line) for line in target.read_text().splitlines()]
    assert [record["page"] for record in records] == [0, 1]
    assert len({record["doc_id"] for record in records}) == 2
    assert [record["text"] for record in records] == ["Source 0", "Source 1"]


def test_mixed_pipeline_keeps_native_text_but_blocks_unrecognized_image_regions(tmp_path):
    from zocr.service.executor import execute_pipeline

    source = tmp_path / "mixed.pdf"
    _make_pdf(source, raster=True)
    out = tmp_path / "out"
    summary = execute_pipeline(timeout_sec=60, inputs=[str(source)], outdir=str(out),
                               do_tune=False, toy_lite=True, sweeps_fixed=1, max_cells=32, k=1)
    records = [json.loads(line) for line in (out / "rag/cells.jsonl").read_text().splitlines()]
    assert records[0]["text"] == "Native evidence 14"
    assert records[1]["text"] == ""
    assert records[1]["meta"]["source_kind"] == "image_region"
    assert records[1]["meta"]["required_ocr_missing"]
    assert summary["pdf_native"]["0"]["export_route"] == "mixed_unrecognized"
    assert summary["publication_status"] == "blocked"
    assert summary["quality_status"] == "fail"


def test_native_japanese_text_stays_unicode_without_ocr(tmp_path):
    source = tmp_path / "japanese.pdf"
    with fitz.open() as doc:
        page = doc.new_page(width=300, height=200)
        page.insert_text((30, 50), "デジタル庁 14", fontname="japan")
        doc.save(source)
    plan = extract_native_page(source, 0, _render(source), tmp_path / "out")
    assert plan["route"] == "native"
    assert "デジタル庁" in "".join(record["text"] for record in plan["records"])


def test_native_page_rejects_a_changed_original(tmp_path):
    source = tmp_path / "native.pdf"
    _make_pdf(source)
    with pytest.raises(ValueError, match="no longer matches"):
        extract_native_page(source, 0, _render(source), tmp_path / "out", expected_sha256="0" * 64)
