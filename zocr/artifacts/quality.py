from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def quality_summary(outdir: str | Path) -> dict[str, Any]:
    path = Path(outdir) / "doc.contextual.jsonl.quality.json"
    status = "unknown"
    if path.is_file():
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            if report.get("schema") == "zocr.export_quality.v1" and report.get("status") in {"pass", "warn", "fail"}:
                status = report["status"]
        except (OSError, ValueError, AttributeError, TypeError):
            pass
    return {
        "quality_status": status,
        "publication_status": "published" if status == "pass" else "blocked" if status == "fail" else "review_required",
    }
