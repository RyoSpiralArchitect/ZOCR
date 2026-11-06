from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

from .manifest import _sha256_file
from ..core.confidence import record_confidence


def _bound_status(path: Path, root: Path, required: str) -> str:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        hashes = report.get("source_hashes")
        if (report.get("schema") not in {"zocr.export_quality.v1", "zocr.final_export_quality.v1"}
                or not isinstance(hashes, dict) or required not in hashes):
            return "unknown"
        for name, expected in hashes.items():
            source = (root / name).resolve()
            if not source.is_relative_to(root.resolve()) or not source.is_file() or _sha256_file(source) != expected:
                return "unknown"
        return report["status"] if report.get("status") in {"pass", "warn", "fail"} else "unknown"
    except (OSError, ValueError, AttributeError, TypeError):
        return "unknown"


def quality_summary(outdir: str | Path) -> dict[str, Any]:
    root = Path(outdir)
    final = root / "final_export.quality.json"
    if final.exists():
        status = _bound_status(final, root, "rag/cells.jsonl")
    elif (root / "doc.mm.jsonl").exists() or (root / "rag/cells.jsonl").exists():
        status = "unknown"
    else:
        status = _bound_status(root / "doc.contextual.jsonl.quality.json", root, "doc.contextual.jsonl")
    return {
        "quality_status": status,
        "publication_status": "published" if status == "pass" else "blocked" if status == "fail" else "review_required",
    }


def write_final_quality(outdir: str | Path, *, min_confidence: float = .58, export_complete: bool = True) -> None:
    root = Path(outdir)
    cells = root / "rag/cells.jsonl"
    if not cells.is_file():
        return
    baseline = _bound_status(root / "doc.contextual.jsonl.quality.json", root, "doc.contextual.jsonl")
    count = uncertain = 0
    with cells.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            meta = record.get("meta") or {}
            confidence = record_confidence(record).get("ocr")
            image = record.get("image_path")
            source = (root / image).resolve() if isinstance(image, str) else None
            count += 1
            uncertain += bool(confidence is None or confidence < min_confidence or meta.get("synthetic")
                              or meta.get("evidence_kind", "observed") != "observed" or meta.get("needs_review")
                              or source is None or not source.is_relative_to(root.resolve()) or not source.is_file())
    augmented = root / "doc.mm.jsonl"
    with augmented.open(encoding="utf-8") if augmented.is_file() else io.StringIO() as stream:
        expected_count = sum(bool(line.strip()) for line in stream)
    status = ("fail" if not export_complete or not count or count != expected_count or baseline == "fail"
              else "warn" if uncertain or baseline == "warn" else baseline)
    hashes = {name: _sha256_file(root / name) for name in (
        "rag/cells.jsonl", "doc.mm.jsonl", "doc.contextual.jsonl",
    ) if (root / name).is_file()}
    report = {"schema": "zocr.final_export_quality.v1", "status": status, "source_hashes": hashes,
              "baseline_status": baseline, "records": count, "expected_records": expected_count,
              "export_complete": export_complete, "uncertain_records": uncertain,
              "minimum_confidence": min_confidence,
              "policy": "Final export diagnostics, not human verification or an access-control gate."}
    (root / "final_export.quality.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
