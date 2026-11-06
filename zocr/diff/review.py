from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import math
from pathlib import Path
from typing import Any

from ..artifacts.quality import quality_summary
from ..core.confidence import record_confidence


def source_evidence(record: dict[str, Any]) -> dict[str, Any]:
    meta = record.get("meta") or {}
    return {
        "doc_id": record.get("doc_id"), "page": record.get("page"),
        "row": record.get("row"), "col": record.get("col"),
        "bbox": record.get("bbox"), "image_path": record.get("image_path") or meta.get("image_path"),
        "trace_id": record.get("trace_id") or meta.get("trace_id") or meta.get("trace"),
        "text": record.get("text"),
        "observed_text": meta.get("observed_text", None if meta.get("synthetic") else record.get("text")),
        "confidence": record_confidence(record),
        "synthetic": bool(meta.get("synthetic")),
        "synthetic_role": meta.get("synthetic_role"),
        "evidence_kind": meta.get("evidence_kind", "inferred" if meta.get("synthetic") else "observed"),
    }


def run_root(cells: Path) -> Path:
    return cells.parent.parent if cells.parent.name == "rag" else cells.parent


def annotate_review(events: list[dict], cells_a: Path, cells_b: Path, *,
                    source_hashes: dict[str, str] | None = None) -> dict[str, Any]:
    hashes = source_hashes or {side: hashlib.sha256(path.read_bytes()).hexdigest() for side, path in (("a", cells_a), ("b", cells_b))}
    qualities = {side: quality_summary(run_root(path))["quality_status"] for side, path in (("a", cells_a), ("b", cells_b))}
    for index, event in enumerate(events):
        event.pop("event_id", None)
        evidence = event.get("evidence", {})
        reasons = []
        low_confidence = False
        for side in ("a", "b"):
            if qualities[side] != "pass":
                reasons.append(f"quality_{side}:{qualities[side]}")
            for cell in evidence.get(side, []):
                confidence = cell.get("confidence")
                if isinstance(confidence, dict):
                    confidence = confidence.get("ocr")
                if not isinstance(confidence, (int, float)) or not math.isfinite(confidence):
                    reasons.append(f"unknown_confidence_{side}")
                elif confidence < .58:
                    low_confidence = True
                    reasons.append(f"low_confidence_{side}")
                if cell.get("evidence_kind") != "observed":
                    reasons.append(f"inferred_evidence_{side}")
                if not cell.get("bbox") or not cell.get("image_path"):
                    reasons.append(f"missing_image_evidence_{side}")
        if not evidence:
            reasons.append("missing_cell_evidence")
        event["review"] = {
            "status": "needs_review",
            "classification": "possible_ocr_change" if low_confidence else "unverified_change",
            "reasons": sorted(set(reasons)),
        }
        identity = json.dumps({"sources": hashes, "index": index,
                               "event": {key: value for key, value in event.items() if key != "review"}},
                              ensure_ascii=False, sort_keys=True)
        event["event_id"] = hashlib.sha256(identity.encode()).hexdigest()
    return {"schema": "zocr.diff_review.v1", "source_hashes": hashes, "source_quality": qualities,
            "pending": len(events), "policy": "Changes require human verification against source evidence."}


def _crop(evidence: dict, source: Path, image_root: Path) -> str | None:
    from PIL import Image

    raw = evidence.get("image_path")
    box = evidence.get("bbox")
    if not raw or not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    path = Path(raw)
    candidates = [path] if path.is_absolute() else [source.parent / path, run_root(source) / path, image_root / path]
    for candidate in candidates:
        candidate = candidate.resolve()
        if not candidate.is_relative_to(image_root.resolve()) or not candidate.is_file():
            continue
        try:
            if candidate.stat().st_size > 32 * 1024 * 1024:
                continue
            coords = [float(v) for v in box]
            if not all(math.isfinite(v) for v in coords):
                continue
            with Image.open(candidate) as image:
                if image.width * image.height > 40_000_000:
                    continue
                x0, y0, x1, y1 = coords
                bounds = (max(0, x0), max(0, y0), min(image.width, x1), min(image.height, y1))
                if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                    continue
                crop = image.crop(bounds).convert("RGB")
                crop.thumbnail((800, 400))
                buffer = io.BytesIO()
                crop.save(buffer, format="PNG")
            return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        except (OSError, ValueError, TypeError, Image.DecompressionBombError):
            continue
    return None


def render_review(result: dict, out_path: Path, *, cells_a: Path, cells_b: Path,
                  image_root: Path | None = None) -> None:
    sections = []
    crop_budget = 100
    for event in result["events"]:
        panels = []
        for side, source in (("a", cells_a), ("b", cells_b)):
            cells = event.get("evidence", {}).get(side, [])
            details = []
            for evidence in cells[:20]:
                image = _crop(evidence, source, image_root or run_root(source)) if crop_budget > 0 else None
                crop_budget -= 1
                picture = f'<img alt="Source crop" src="{image}">' if image else '<p>Source crop unavailable; verify the source location below.</p>'
                details.append(picture + '<pre>' + html.escape(json.dumps(evidence, ensure_ascii=False, indent=2)) + '</pre>')
            panels.append(f'<section><h3>Version {side.upper()}</h3>' + ''.join(details) + '</section>')
        event_id = html.escape(event["event_id"], quote=True)
        event_details = {key: value for key, value in event.items() if key != "evidence"}
        sections.append(
            f'<article data-event="{event_id}"><h2>{html.escape(str(event.get("type")))}</h2>'
            f'<pre>{html.escape(json.dumps(event_details, ensure_ascii=False, indent=2))}</pre>'
            f'<div class="pair">{"".join(panels)}</div>'
            '<label>Decision <select><option value="needs_review">Needs review</option>'
            '<option value="approved">Verified change</option><option value="rejected">Reject / OCR error</option></select></label>'
            '<label>Reason <textarea></textarea></label></article>'
        )
    payload = json.dumps(result.get("review", {}), ensure_ascii=False).replace("<", "\\u003c")
    document = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'">
<title>ZOCR Evidence Review</title><style>
body{font:16px system-ui;margin:2rem;color:#202530;background:#f5f7fa}article{background:white;padding:1rem;margin:1rem 0;border:1px solid #ccd}
.pair{display:flex;gap:1rem}.pair section{flex:1;min-width:0}pre{white-space:pre-wrap;overflow-wrap:anywhere}img{max-width:100%;border:1px solid #ccd}
label{display:block;margin:.5rem 0}textarea{display:block;width:95%;min-height:4rem}button{padding:.7rem} @media(max-width:700px){.pair{display:block}}
</style><h1>Evidence-based change review</h1>
<p>OCR differences are not confirmed document changes. Inspect both sources before approving. Decisions stay in this browser until exported; they do not publish or modify source artifacts.</p>
<label>Reviewer <input id="reviewer" required></label><button id="export">Export review decisions</button>
'''
    document += '<pre>' + html.escape(json.dumps(result.get("review", {}), indent=2)) + '</pre>'
    document += ''.join(sections) or '<p>No changes detected in the validated inputs.</p>'
    document += '<script type="application/json" id="review-data">' + payload + '</script>'
    document += '''<script>
document.getElementById('export').onclick=()=>{
 const reviewer=document.getElementById('reviewer').value.trim();
 if(!reviewer){alert('Enter a reviewer name.');return;}
 const decisions=Array.from(document.querySelectorAll('article')).map(e=>({event_id:e.dataset.event,status:e.querySelector('select').value,reason:e.querySelector('textarea').value}));
 if(decisions.some(d=>d.status!=='needs_review'&&!d.reason.trim())){alert('Add a reason for each reviewed decision.');return;}
 const bundle={schema:'zocr.diff_decisions.v1',reviewer,reviewed_at:new Date().toISOString(),...JSON.parse(document.getElementById('review-data').textContent),decisions};
 bundle.schema='zocr.diff_decisions.v1';
 const url=URL.createObjectURL(new Blob([JSON.stringify(bundle,null,2)],{type:'application/json'}));
 const a=document.createElement('a');a.href=url;a.download='review-decisions.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
</script></html>'''
    out_path.write_text(document, encoding="utf-8")
