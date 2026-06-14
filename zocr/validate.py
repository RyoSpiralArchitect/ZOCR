from __future__ import annotations

import argparse
import hashlib
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


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _guess_kind(path: Path) -> str:
    if path.is_dir():
        return "dir"
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".zip":
        return "zip"
    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        return "image"
    if suffix == ".csv":
        return "csv"
    if suffix == ".pkl":
        return "pickle"
    if suffix in {".html", ".htm"}:
        return "html"
    if suffix == ".md":
        return "markdown"
    return "file"


def _resolve_artifact_path(base_dir: Path, rel: str) -> Tuple[Path, bool]:
    raw = Path(rel)
    path = raw if raw.is_absolute() else base_dir / raw
    try:
        resolved = path.resolve()
        base = base_dir.resolve()
    except Exception:
        return path, True
    return resolved, resolved == base or base in resolved.parents


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


def _validate_simple_summary(obj: Any) -> List[str]:
    errors: List[str] = []
    if not isinstance(obj, dict):
        return ["summary.json: expected object"]
    schema = obj.get("schema")
    if schema is None:
        return errors
    if schema != "zocr.simple_run.v1":
        errors.append(f"summary.json: unexpected schema={obj.get('schema')!r}")
    totals = obj.get("totals")
    if not isinstance(totals, dict):
        errors.append("summary.json: missing/invalid totals")
    return errors


def _validate_simple_manifest(obj: Any) -> List[str]:
    errors: List[str] = []
    if not isinstance(obj, dict):
        return ["manifest.json: expected object"]
    schema = obj.get("schema")
    if schema is None:
        return errors
    if schema != "zocr.simple_run.v1":
        errors.append(f"manifest.json: unexpected schema={obj.get('schema')!r}")
    artifacts = obj.get("artifacts")
    if not isinstance(artifacts, dict):
        errors.append("manifest.json: missing/invalid artifacts")
    return errors


def _validate_jsonl(path: Path, *, max_lines: int) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= max_lines:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception as exc:
                    errors.append(f"{path.name}: invalid JSONL at line {idx+1}: {exc}")
                    break
                if not isinstance(payload, dict):
                    warnings.append(f"{path.name}: JSONL line {idx+1} is not an object")
                    break
    except FileNotFoundError:
        errors.append(f"{path.name}: missing")
    except Exception as exc:
        errors.append(f"{path.name}: unreadable: {exc}")
    return errors, warnings


def _validate_artifact_integrity(name: str, entry: Dict[str, Any], path: Path) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    kind = entry.get("kind")
    actual_kind = _guess_kind(path)
    if isinstance(kind, str) and kind and kind != actual_kind:
        warnings.append(f"manifest: artifact {name!r} kind={kind!r} but path looks like {actual_kind!r}")
    if not path.is_file():
        return errors, warnings
    try:
        actual_bytes = path.stat().st_size
    except Exception as exc:
        errors.append(f"{entry.get('path')}: cannot stat artifact: {exc}")
        return errors, warnings
    expected_bytes = entry.get("bytes")
    if isinstance(expected_bytes, int) and expected_bytes != actual_bytes:
        errors.append(f"{entry.get('path')}: byte size mismatch manifest={expected_bytes} actual={actual_bytes}")
    expected_sha = entry.get("sha256")
    if isinstance(expected_sha, str) and expected_sha:
        try:
            actual_sha = _sha256_file(path)
        except Exception as exc:
            errors.append(f"{entry.get('path')}: sha256 check failed: {exc}")
        else:
            if actual_sha != expected_sha:
                errors.append(f"{entry.get('path')}: sha256 mismatch")
    return errors, warnings


def _validate_manifest_audit(manifest: Dict[str, Any]) -> List[str]:
    audit = manifest.get("audit")
    if not isinstance(audit, dict):
        return []
    warnings: List[str] = []
    audit_warnings = audit.get("warnings")
    if isinstance(audit_warnings, list):
        for item in audit_warnings:
            if isinstance(item, str) and item:
                warnings.append(f"manifest audit: {item}")
    missing = audit.get("missing_references")
    if isinstance(missing, list) and missing:
        warnings.append(f"manifest audit: {len(missing)} missing artifact reference(s)")
    return warnings


def _validate_manifest(manifest: Any, *, base_dir: Path, max_jsonl_lines: int) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    if not isinstance(manifest, dict):
        return (["manifest: expected object"], warnings)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        errors.append(f"manifest: unexpected schema={manifest.get('schema')!r}")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        errors.append(f"manifest: unexpected schema_version={manifest.get('schema_version')!r}")
    warnings.extend(_validate_manifest_audit(manifest))

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
        path, inside_base = _resolve_artifact_path(base_dir, rel)
        if not inside_base:
            errors.append(f"artifact path escapes run directory: {rel}")
            continue
        if not path.exists():
            errors.append(f"artifact missing: {rel}")
            continue
        integrity_errors, integrity_warnings = _validate_artifact_integrity(name, entry, path)
        errors.extend(integrity_errors)
        warnings.extend(integrity_warnings)
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
            elif name == "simple_summary":
                errors.extend(_validate_simple_summary(obj))
            elif name == "simple_manifest":
                errors.extend(_validate_simple_manifest(obj))
        elif kind == "jsonl":
            jsonl_errors, jsonl_warnings = _validate_jsonl(path, max_lines=max_jsonl_lines)
            errors.extend(jsonl_errors)
            warnings.extend(jsonl_warnings)
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
