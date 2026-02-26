from __future__ import annotations

import json
import subprocess
import sys


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

