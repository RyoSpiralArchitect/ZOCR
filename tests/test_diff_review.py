import json

import pytest

from zocr.diff.differ import SemanticDiffer, load_jsonl, to_cell


def write_cells(path, value, confidence=.9):
    record = {
        "doc_id": "invoice", "page": 1, "table_index": 0, "row": 1, "col": 0,
        "header": "Amount", "text": str(value), "bbox": [0, 0, 50, 30],
        "image_path": "page.png", "meta": {"confidence": confidence, "filters": {"amount": value}},
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return record


def test_diff_preserves_nested_filters_and_zero():
    cell = to_cell({"text": "USD 0", "meta": {"filters": {"amount": 0}}})
    assert cell.filters["amount"] == 0


@pytest.mark.parametrize("content", ['{"text": "ok"}\ninvalid\n', '[]\n'])
def test_corrupted_diff_input_is_not_silently_ignored(tmp_path, content):
    path = tmp_path / "cells.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_jsonl(path)


def test_missing_diff_input_is_not_an_empty_document(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_jsonl(tmp_path / "missing.jsonl")


def test_cell_change_carries_original_evidence(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_cells(a, 0)
    write_cells(b, 120, confidence=.2)
    result = SemanticDiffer().compare_bundle(a, b)
    event = next(e for e in result["events"] if e["type"] == "cell_updated")
    assert event["numeric_delta"] == 120
    assert event["evidence"]["a"][0]["bbox"] == [0, 0, 50, 30]
    assert event["evidence"]["b"][0]["confidence"] == {"ocr": .2}
    assert event["review"]["classification"] == "possible_ocr_change"
    assert event["review"]["status"] == "needs_review"
    assert event["event_id"]
    assert result["review"]["source_hashes"]["a"]
    assert result == SemanticDiffer().compare_bundle(a, b)


def test_quality_status_controls_publication_recommendation(tmp_path):
    from zocr.artifacts.quality import quality_summary

    assert quality_summary(tmp_path)["publication_status"] == "review_required"
    import hashlib
    source = tmp_path / "doc.contextual.jsonl"
    source.write_text('{}\n', encoding="utf-8")
    hashes = {source.name: hashlib.sha256(source.read_bytes()).hexdigest()}
    path = tmp_path / "doc.contextual.jsonl.quality.json"
    for status, publication in (("pass", "published"), ("warn", "review_required"), ("fail", "blocked")):
        path.write_text(json.dumps({"schema": "zocr.export_quality.v1", "status": status, "source_hashes": hashes}), encoding="utf-8")
        assert quality_summary(tmp_path) == {"quality_status": status, "publication_status": publication}
    path.write_text('invalid', encoding="utf-8")
    assert quality_summary(tmp_path)["quality_status"] == "unknown"


def test_review_embeds_crops_and_escapes_untrusted_text(tmp_path):
    from PIL import Image
    from zocr.diff.review import render_review

    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_cells(a, 10)
    record = write_cells(b, 20)
    record["text"] = "</script><script>alert(1)</script>"
    b.write_text(json.dumps(record) + "\n", encoding="utf-8")
    Image.new("RGB", (60, 40), "white").save(tmp_path / "page.png")
    result = SemanticDiffer().compare_bundle(a, b)
    output = tmp_path / "review.html"
    render_review(result, output, cells_a=a, cells_b=b)
    page = output.read_text(encoding="utf-8")
    assert "data:image/png;base64," in page
    assert "Export review decisions" in page
    assert "</script><script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page


def test_review_never_reads_images_outside_allowed_root(tmp_path):
    from PIL import Image
    from zocr.diff.review import _crop

    allowed = tmp_path / "allowed"
    allowed.mkdir()
    private = tmp_path / "outside.png"
    Image.new("RGB", (20, 20)).save(private)
    evidence = {"image_path": str(private), "bbox": [0, 0, 10, 10]}
    assert _crop(evidence, allowed / "cells.jsonl", allowed) is None
    (allowed / "link.png").symlink_to(private)
    evidence["image_path"] = "link.png"
    assert _crop(evidence, allowed / "cells.jsonl", allowed) is None


def test_tables_from_distinct_documents_do_not_collapse():
    from zocr.diff.differ import build_tables

    cells = [to_cell({"doc_id": doc, "page": 1, "table_index": 0,
                      "row": 1, "col": 0, "text": "value"}) for doc in ("a", "b")]
    assert len(build_tables(cells)) == 2


def test_row_reordering_is_not_a_value_change(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    records = [{"page": 1, "table_index": 0, "row": row, "col": col,
                "header": ["ID", "Amount"][col], "text": text}
               for row, values in enumerate((("A", "10"), ("B", "20")), start=1)
               for col, text in enumerate(values)]
    a.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    b.write_text("\n".join(json.dumps({**r, "row": 3-r["row"]}) for r in records), encoding="utf-8")
    result = SemanticDiffer().compare_bundle(a, b)
    assert not result["events"]
