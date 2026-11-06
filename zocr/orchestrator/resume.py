from __future__ import annotations

import hashlib
import glob
from importlib import metadata
import json
import os
from pathlib import Path
from typing import Any


from ..utils.formats import IMAGE_EXTENSIONS


STEP_OUTPUTS = {
    "OCR": "doc.zocr.json",
    "Export": "doc.contextual.jsonl",
    "Augment": "doc.mm.jsonl",
    "Index": "bm25.pkl",
    "Monitor": "monitor.csv",
}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_resume(outdir: str, options: dict[str, Any]) -> str:
    options = {k: v for k, v in options.items() if k not in {"resume", "outdir", "print_stage_trace"}}
    inputs = {}
    for value in options.values():
        for item in value if isinstance(value, list) else [value]:
            if not isinstance(item, str):
                continue
            paths = glob.glob(item, recursive=True) if glob.has_magic(item) else [item]
            for candidate in paths:
                path = Path(candidate)
                if path.is_file():
                    inputs[str(path.resolve())] = file_hash(path)
                elif path.is_dir():
                    for child in sorted(path.rglob("*")):
                        if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS | {".pdf"}:
                            inputs[str(child.resolve())] = file_hash(child)
    code_root = Path(__file__).resolve().parents[1]
    code = {str(path.relative_to(code_root)): file_hash(path) for path in sorted(code_root.rglob("*.py"))}
    environment = {k: v for k, v in os.environ.items() if k.startswith("ZOCR_")}
    versions = {}
    for package in ("numpy", "Pillow", "pypdfium2", "pytesseract", "easyocr", "pdfminer.six", "PyMuPDF"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    identity = {"options": options, "inputs": inputs, "code": code, "environment": environment,
                "engine_versions": versions}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()
    path = Path(outdir) / "resume_identity.json"
    path.write_text(json.dumps({"fingerprint": signature}) + "\n", encoding="utf-8")
    return signature


def step_integrity(outdir: str, name: str) -> dict[str, str]:
    filename = STEP_OUTPUTS.get(name)
    path = Path(outdir) / filename if filename else None
    if path is None or not path.is_file():
        return {}
    return {filename: file_hash(path)}


def reusable_steps(outdir: str, signature: str) -> set[str]:
    history = Path(outdir) / "pipeline_history.jsonl"
    if not history.is_file():
        return set()
    latest = {}
    for line in history.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("run_fingerprint") == signature:
            latest[record.get("name") or record.get("step")] = record
    reusable = set()
    for name in STEP_OUTPUTS:
        record = latest.get(name, {})
        hashes = record.get("output_hashes")
        if not record.get("ok") or not hashes or hashes != step_integrity(outdir, name):
            break
        reusable.add(name)
    return reusable
