from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .artifacts.manifest import (
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA,
    MANIFEST_SCHEMA_VERSION,
    build_manifest,
    write_manifest,
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_doc_zocr(obj: Any) -> List[str]:
    errors: List[str] = []
    if not isinstance(obj, dict):
        return ["doc.zocr.json: expected object"]
    if not isinstance(obj.get("doc_id"), str) or not obj.get("doc_id"):
        errors.append("doc.zocr.json: missing/invalid doc_id")
    pages = obj.get("pages")
    if not isinstance(pages, list):
        errors.append("doc.zocr.json: missing/invalid pages[]")
    else:
        for idx, page in enumerate(pages[:3]):
            if not isinstance(page, dict):
                errors.append(f"doc.zocr.json: pages[{idx}] must be object")
                continue
            if "index" in page and not isinstance(page.get("index"), int):
                errors.append(f"doc.zocr.json: pages[{idx}].index must be int")
    return errors


def _validate_pipeline_summary(obj: Any) -> List[str]:
    errors: List[str] = []
    if not isinstance(obj, dict):
        return ["pipeline_summary.json: expected object"]
    inputs = obj.get("inputs")
    if inputs is not None and not isinstance(inputs, list):
        errors.append("pipeline_summary.json: inputs must be list if present")
    if "generated_at" in obj and not isinstance(obj.get("generated_at"), str):
        errors.append("pipeline_summary.json: generated_at must be str if present")
    return errors


def _validate_jsonl(path: Path, *, max_lines: int) -> List[str]:
    errors: List[str] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= max_lines:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)
                except Exception as exc:
                    errors.append(f"{path.name}: invalid JSONL at line {idx+1}: {exc}")
                    break
    except FileNotFoundError:
        errors.append(f"{path.name}: missing")
    except Exception as exc:
        errors.append(f"{path.name}: unreadable: {exc}")
    return errors


def _validate_manifest(manifest: Any, *, base_dir: Path, max_jsonl_lines: int) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    if not isinstance(manifest, dict):
        return (["manifest: expected object"], warnings)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        errors.append(f"manifest: unexpected schema={manifest.get('schema')!r}")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        errors.append(f"manifest: unexpected schema_version={manifest.get('schema_version')!r}")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        errors.append("manifest: artifacts must be object")
        return (errors, warnings)

    for name, entry in artifacts.items():
        if not isinstance(entry, dict):
            warnings.append(f"manifest: artifact {name!r} must be object")
            continue
        rel = entry.get("path")
        if not isinstance(rel, str) or not rel:
            warnings.append(f"manifest: artifact {name!r} missing path")
            continue
        path = base_dir / rel
        if not path.exists():
            errors.append(f"artifact missing: {rel}")
            continue
        kind = entry.get("kind")
        if kind == "json":
            try:
                obj = _load_json(path)
            except Exception as exc:
                errors.append(f"{rel}: invalid json: {exc}")
                continue
            if name == "doc_zocr":
                errors.extend(_validate_doc_zocr(obj))
            elif name == "pipeline_summary":
                errors.extend(_validate_pipeline_summary(obj))
        elif kind == "jsonl":
            errors.extend(_validate_jsonl(path, max_lines=max_jsonl_lines))
    return (errors, warnings)


def _discover_outdir_from_path(path: Path) -> Path:
    if path.is_dir():
        return path
    return path.parent


def validate_path(path: Path, *, strict: bool, max_jsonl_lines: int, write_manifest_flag: bool) -> int:
    outdir = _discover_outdir_from_path(path)
    if not outdir.exists():
        print(f"[ERROR] missing path: {path}", file=sys.stderr)
        return 2

    manifest_path = outdir / MANIFEST_FILENAME
    manifest_obj: Optional[Dict[str, Any]] = None
    if manifest_path.exists():
        try:
            loaded = _load_json(manifest_path)
        except Exception as exc:
            print(f"[ERROR] invalid manifest json: {manifest_path}: {exc}", file=sys.stderr)
            return 2
        if isinstance(loaded, dict):
            manifest_obj = loaded
        else:
            print(f"[ERROR] invalid manifest type: {manifest_path}", file=sys.stderr)
            return 2
    else:
        summary_path = outdir / "pipeline_summary.json"
        summary_obj = None
        if summary_path.exists():
            try:
                summary_obj = _load_json(summary_path)
            except Exception:
                summary_obj = None
        manifest_obj = build_manifest(outdir, summary=summary_obj if isinstance(summary_obj, dict) else None)
        if write_manifest_flag:
            write_manifest(outdir, summary=summary_obj if isinstance(summary_obj, dict) else None)

    errors, warnings = _validate_manifest(manifest_obj, base_dir=outdir, max_jsonl_lines=max_jsonl_lines)
    if warnings:
        for w in warnings:
            print(f"[WARN] {outdir}: {w}", file=sys.stderr)
    if errors:
        for e in errors:
            print(f"[ERROR] {outdir}: {e}", file=sys.stderr)
        return 1
    if strict and warnings:
        return 1
    return 0


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser("zocr validate")
    parser.add_argument("path", nargs="*", default=["."], help="Run directory (or any file inside it).")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors.")
    parser.add_argument("--max-jsonl-lines", type=int, default=1000, help="Max JSONL lines to parse.")
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help=f"Write {MANIFEST_FILENAME} if missing.",
    )
    args = parser.parse_args(argv)

    rc = 0
    for raw in args.path:
        rc = max(
            rc,
            validate_path(
                Path(raw),
                strict=bool(args.strict),
                max_jsonl_lines=max(0, int(args.max_jsonl_lines)),
                write_manifest_flag=bool(args.write_manifest),
            ),
        )
    raise SystemExit(rc)


if __name__ == "__main__":  # pragma: no cover
    main()

