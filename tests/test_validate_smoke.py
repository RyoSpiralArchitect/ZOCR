from __future__ import annotations

import json
import subprocess
import sys

from zocr.artifacts.manifest import build_manifest, write_manifest


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_validate_writes_manifest(tmp_path) -> None:
    outdir = tmp_path / "out"
    outdir.mkdir()
    (outdir / "doc.zocr.json").write_text(
        json.dumps({"doc_id": "doc", "pages": [], "metrics": {}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (outdir / "pipeline_summary.json").write_text(
        json.dumps({"inputs": [], "generated_at": "2026-01-01T00:00:00Z"}, ensure_ascii=False),
        encoding="utf-8",
    )

    proc = _run("-m", "zocr", "validate", str(outdir), "--write-manifest")
    assert proc.returncode == 0, proc.stderr
    assert (outdir / "zocr.manifest.json").exists()


def test_manifest_discovers_simple_handoff_artifacts(tmp_path) -> None:
    outdir = tmp_path / "simple"
    outdir.mkdir()
    (outdir / "pages.json").write_text(json.dumps([{"page_index": 0}], ensure_ascii=False), encoding="utf-8")
    (outdir / "summary.json").write_text(
        json.dumps(
            {
                "schema": "zocr.simple_run.v1",
                "totals": {"documents": 1, "pages": 1, "regions": 1},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (outdir / "regions.jsonl").write_text(json.dumps({"type": "text"}, ensure_ascii=False) + "\n", encoding="utf-8")
    (outdir / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "zocr.simple_run.v1",
                "artifacts": {"pages_json": "pages.json"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    manifest = build_manifest(outdir)

    assert manifest["audit"]["artifact_count"] == 4
    assert manifest["audit"]["artifact_kinds"]["json"] == 3
    assert manifest["artifacts"]["simple_regions"]["kind"] == "jsonl"
    assert manifest["artifacts"]["simple_regions"]["bytes"] > 0
    assert len(manifest["artifacts"]["simple_regions"]["sha256"]) == 64


def test_validate_detects_stale_manifest_hash(tmp_path) -> None:
    outdir = tmp_path / "out"
    outdir.mkdir()
    doc_path = outdir / "doc.zocr.json"
    doc_path.write_text(
        json.dumps({"doc_id": "doc", "pages": []}, ensure_ascii=False),
        encoding="utf-8",
    )
    write_manifest(outdir)
    doc_path.write_text(
        json.dumps({"doc_id": "doc", "pages": [], "mutated": True}, ensure_ascii=False),
        encoding="utf-8",
    )

    proc = _run("-m", "zocr", "validate", str(outdir))

    assert proc.returncode == 1
    assert "sha256 mismatch" in proc.stderr
