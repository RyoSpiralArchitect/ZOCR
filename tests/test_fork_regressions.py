import json
import os
import subprocess
import sys

import pytest

from zocr.diff.differ import SemanticDiffer, to_cell
from zocr.diff.review import annotate_review
from zocr.orchestrator.resume import prepare_resume


def test_identical_multidocument_tables_across_hash_seeds(tmp_path):
    path = tmp_path / "cells.jsonl"
    records = [{"doc_id": doc, "page": 1, "table_index": 0, "row": 1, "col": col,
                "header": ["ID", "Amount"][col], "text": value}
               for doc, amount in (("x", "10"), ("y", "99"))
               for col, value in enumerate(("same", amount))]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    code = "from pathlib import Path;from zocr.diff.differ import SemanticDiffer;import sys;assert not SemanticDiffer().compare_bundle(Path(sys.argv[1]),Path(sys.argv[1]))['events']"
    for seed in ("1", "7", "13"):
        result = subprocess.run([sys.executable, "-c", code, str(path)],
                                env={**os.environ, "PYTHONHASHSEED": seed}, capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("suffix", [".bmp", ".webp"])
def test_supported_directory_inputs_invalidate_resume(tmp_path, suffix):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    page = inputs / ("page" + suffix)
    page.write_bytes(b"old")
    options = {"inputs": [str(inputs)]}
    first = prepare_resume(str(tmp_path), options)
    page.write_bytes(b"new")
    assert first != prepare_resume(str(tmp_path), options)
    added = inputs / ("another" + suffix)
    added.write_bytes(b"new")
    second = prepare_resume(str(tmp_path), options)
    added.unlink()
    assert second != prepare_resume(str(tmp_path), options)


def test_empty_cells_and_invalid_coordinates_are_not_validated_changes(tmp_path):
    path = tmp_path / "cells.jsonl"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        SemanticDiffer().compare_bundle(path, path)
    path.write_text(json.dumps({"page": None, "text": "value"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"cells.jsonl:1"):
        SemanticDiffer().compare_bundle(path, path)


def test_invalid_numeric_filters_fall_back_and_trace_survives():
    a = to_cell({"text": "10", "meta": {"filters": {"amount": ""}, "trace": "source"}})
    b = to_cell({"text": "20", "meta": {"filters": {"amount": False}}})
    event = SemanticDiffer().compare_cells(a, b)
    assert event["numeric_delta"] == 10
    assert event["evidence"]["a"][0]["trace_id"] == "source"


def test_duplicate_events_have_distinct_stable_ids(tmp_path):
    path = tmp_path / "cells.jsonl"
    path.write_text('{}\n', encoding="utf-8")
    events = [{"type": "section_added", "title": "same", "page": 1} for _ in range(2)]
    annotate_review(events, path, path)
    ids = [event["event_id"] for event in events]
    assert len(set(ids)) == 2
    annotate_review(events, path, path)
    assert ids == [event["event_id"] for event in events]


def test_quality_report_cannot_certify_another_export(tmp_path):
    from zocr.artifacts.quality import quality_summary

    (tmp_path / "doc.contextual.jsonl.quality.json").write_text(
        json.dumps({"schema": "zocr.export_quality.v1", "status": "pass"}), encoding="utf-8")
    (tmp_path / "doc.mm.jsonl").write_text('{"text":"inferred result"}\n', encoding="utf-8")
    assert quality_summary(tmp_path)["quality_status"] == "unknown"


def test_final_quality_is_bound_to_the_distributed_cells(tmp_path):
    import hashlib
    from PIL import Image
    from zocr.artifacts.evidence import archive_page_image
    from zocr.artifacts.quality import quality_summary, write_final_quality

    image = archive_page_image(Image.new("RGB", (20, 20), "white"), tmp_path)
    record = {"text": "observed", "image_path": image, "meta": {"confidence": .95}}
    content = json.dumps(record) + "\n"
    source = tmp_path / "doc.contextual.jsonl"
    source.write_text(content, encoding="utf-8")
    (tmp_path / "doc.contextual.jsonl.quality.json").write_text(json.dumps({
        "schema": "zocr.export_quality.v1", "status": "pass",
        "source_hashes": {source.name: hashlib.sha256(source.read_bytes()).hexdigest()},
    }), encoding="utf-8")
    (tmp_path / "doc.mm.jsonl").write_text(content, encoding="utf-8")
    (tmp_path / "rag").mkdir()
    cells = tmp_path / "rag/cells.jsonl"
    cells.write_text(content, encoding="utf-8")
    write_final_quality(tmp_path)
    assert quality_summary(tmp_path)["quality_status"] == "pass"
    cells.write_text('{"text":"changed"}\n', encoding="utf-8")
    assert quality_summary(tmp_path)["quality_status"] == "unknown"
    cells.write_text(content, encoding="utf-8")
    write_final_quality(tmp_path, export_complete=False)
    assert quality_summary(tmp_path)["quality_status"] == "fail"
    cells.write_text(content * 2, encoding="utf-8")
    write_final_quality(tmp_path)
    assert quality_summary(tmp_path)["quality_status"] == "fail"


def test_archived_images_repair_interrupted_writes(tmp_path):
    from pathlib import Path
    from PIL import Image
    from zocr.artifacts.evidence import archive_page_image

    image = Image.new("RGB", (20, 20), "white")
    target = Path(archive_page_image(image, tmp_path))
    expected = target.read_bytes()
    target.write_bytes(b"interrupted")
    assert archive_page_image(image, tmp_path) == str(target)
    assert target.read_bytes() == expected
    assert len(list(target.parent.iterdir())) == 1


def test_standalone_rag_preserves_source_image_base(tmp_path):
    from pathlib import Path
    from PIL import Image
    from zocr.artifacts.evidence import archive_page_image
    from zocr.core.exporters import export_rag_bundle

    image = archive_page_image(Image.new("RGB", (20, 20), "white"), tmp_path)
    source = tmp_path / "doc.mm.jsonl"
    source.write_text(json.dumps({"doc_id": "doc", "page": 1, "table_index": 0, "text": "value",
                                  "image_path": image}) + "\n", encoding="utf-8")
    exported = export_rag_bundle(str(source), str(tmp_path / "standalone"))
    record = json.loads(Path(exported["cells"]).read_text(encoding="utf-8"))
    assert record["image_path"] == image
