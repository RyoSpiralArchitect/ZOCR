"""Standalone CLI entrypoints for the consensus runtime."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from .local_search import (
    build_local_index,
    describe_local_index,
    ensure_local_index,
    query_local,
)
from .runtime import (
    Pipeline,
    auto_calibrate_params,
    autotune_params,
    configure_toy_runtime,
    ensure_dir,
    export_jsonl_with_ocr,
    pdf_to_images_via_poppler,
)

__all__ = [
    "_AUTOCALIB_DEFAULT_SAMPLES",
    "_AUTOTUNE_DEFAULT_TRIALS",
    "_DEMO_INPUT_SUFFIXES",
    "_discover_demo_inputs_for_consensus",
    "_dedup_input_paths",
    "_positive_cli_value",
    "_patch_cli_for_export_and_search",
    "cli_export",
    "cli_index",
    "cli_stats",
    "cli_query",
    "cli_toy_bench",
    "main",
]

_AUTOCALIB_DEFAULT_SAMPLES = 3
_AUTOTUNE_DEFAULT_TRIALS = 6

_DEMO_INPUT_SUFFIXES: Tuple[str, ...] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".gif",
    ".pdf",
)


def _discover_demo_inputs_for_consensus() -> List[str]:
    """Locate demo input files for the standalone consensus CLI."""

    env_override = os.environ.get("ZOCR_DEMO_INPUTS")
    candidate_roots: List[Path] = []
    seen_candidates: Set[str] = set()

    def _add_candidate(path: Path) -> None:
        try:
            resolved = path if path.is_absolute() else (Path.cwd() / path)
            resolved = resolved.resolve()
        except Exception:
            resolved = path
        key = str(resolved)
        if key in seen_candidates:
            return
        seen_candidates.add(key)
        if resolved.exists():
            candidate_roots.append(resolved)

    if env_override:
        for chunk in env_override.split(os.pathsep):
            chunk = chunk.strip()
            if chunk:
                _add_candidate(Path(chunk))

    here = Path(__file__).resolve()
    repo_root = here.parents[2] if len(here.parents) >= 3 else here.parent
    search_roots = [Path.cwd(), repo_root]
    rel_candidates = [
        Path("samples/demo_inputs"),
        Path("samples/input_demo"),
        Path("demo_inputs"),
        Path("input_demo"),
    ]

    for root in search_roots:
        for rel in rel_candidates:
            _add_candidate(root / rel)

    files: List[str] = []
    seen_files: Set[str] = set()

    def _add_file(path: Path) -> None:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        key = str(resolved)
        if key in seen_files:
            return
        if resolved.is_file() and resolved.suffix.lower() in _DEMO_INPUT_SUFFIXES:
            seen_files.add(key)
            files.append(key)

    for candidate in candidate_roots:
        if candidate.is_file():
            _add_file(candidate)
        elif candidate.is_dir():
            for entry in candidate.rglob("*"):
                if entry.is_file() and entry.suffix.lower() in _DEMO_INPUT_SUFFIXES:
                    _add_file(entry)

    return files


def _dedup_input_paths(paths: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    result: List[str] = []
    for path in paths:
        norm = os.path.abspath(path)
        if norm in seen:
            continue
        seen.add(norm)
        result.append(path)
    return result


def _positive_cli_value(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    return value if value > 0 else None


def cli_export(args):
    # Determine source image for current run
    # If --demo used earlier, try demo_inv.png; else use first input
    out_dir = args.out
    src = None
    if args.input:
        src = args.input[0]
    else:
        cand = os.path.join(out_dir, "demo_inv.png")
        src = cand if os.path.exists(cand) else None
    if not src:
        print("No source image found for export.")
        return
    jpath = os.path.join(out_dir, "doc.zocr.json")
    if not os.path.exists(jpath):
        print("doc.zocr.json not found in", out_dir)
        return
    out_jsonl = os.path.join(out_dir, "doc.contextual.jsonl")
    source_images: Union[str, Sequence[str]]
    if len(args.input) > 1:
        source_images = [p for p in args.input if isinstance(p, str)]
    else:
        source_images = src
    runtime_overrides: Dict[str, Any] = {}
    sweeps = getattr(args, "toy_sweeps", None)
    if sweeps is not None and sweeps > 0:
        runtime_overrides["sweeps"] = sweeps
    force_numeric = getattr(args, "force_numeric", None)
    if force_numeric is not None:
        runtime_overrides["force_numeric"] = force_numeric
    if runtime_overrides:
        configure_toy_runtime(**runtime_overrides)
    n = export_jsonl_with_ocr(jpath, source_images, out_jsonl, ocr_engine="toy", contextual=True)
    print("Exported", n, "records to", out_jsonl)


def cli_index(args):
    jsonl = os.path.join(args.out, "doc.contextual.jsonl")
    if not os.path.exists(jsonl):
        print("contextual JSONL not found:", jsonl)
        return
    pkl = os.path.join(args.out, "bm25.pkl")
    index = build_local_index(jsonl, pkl)
    print(f"Wrote local index with {index.total_docs} records -> {pkl}")


def cli_stats(args):
    out_dir = args.out
    jsonl = os.path.join(out_dir, "doc.contextual.jsonl")
    pkl = os.path.join(out_dir, "bm25.pkl")
    if not os.path.exists(pkl):
        print("Missing BM25 index:", pkl)
        return
    stats = describe_local_index(pkl, jsonl_path=jsonl if os.path.exists(jsonl) else None)
    if args.json:
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return
    print(f"Index path: {stats.get('index_path')}")
    if stats.get("jsonl_path"):
        print(f"Contextual JSONL: {stats.get('jsonl_path')}")
    doc_count = stats.get("document_count")
    vocab = stats.get("vocab_size")
    avg_len = stats.get("avg_doc_len")
    print(
        "Documents: {doc_count}  |  Vocabulary: {vocab}  |  Avg length: {avg:.1f}".format(
            doc_count=doc_count if doc_count is not None else "?",
            vocab=vocab if vocab is not None else "?",
            avg=float(avg_len or 0.0),
        )
    )
    if stats.get("median_doc_len") is not None:
        print(
            "Median length: {med:.1f}  |  P95 length: {p95:.1f}  |  Max length: {mx}".format(
                med=float(stats.get("median_doc_len") or 0.0),
                p95=float(stats.get("p95_doc_len") or 0.0),
                mx=int(stats.get("max_doc_len") or 0),
            )
        )
    created = stats.get("created_at")
    if created:
        print("Built at:", created)
    if stats.get("source_path"):
        stale_flag = stats.get("stale")
        stale_txt = " (stale vs JSONL)" if stale_flag else ""
        print(f"Source: {stats['source_path']}{stale_txt}")
    if stats.get("source_missing"):
        print("⚠️  Source JSONL no longer exists; rebuild recommended.")


def cli_query(args):
    out_dir = args.out
    jsonl = os.path.join(out_dir, "doc.contextual.jsonl")
    pkl = os.path.join(out_dir, "bm25.pkl")
    if not os.path.exists(jsonl):
        print("Missing contextual JSONL:", jsonl)
        return
    index, rebuilt = ensure_local_index(jsonl, pkl)
    if rebuilt:
        print("Rebuilt BM25 index from contextual JSONL.")
    merged = query_local(
        jsonl,
        pkl,
        text_query=args.query or "",
        image_query_path=(args.image_query or None),
        topk=args.topk,
        autobuild=False,
        index=index,
    )
    # Print concise results
    for i, r in enumerate(merged, 1):
        ob = r["obj"]
        print(
            f"{i:2d}. score={r['score']:.4f} page={ob.get('page')} row={ob.get('row')} "
            f"col={ob.get('col')} text='{(ob.get('text') or '')[:40]}' bbox={ob.get('bbox')}"
        )


def _normalize_bench_text(text: str, mode: str) -> str:
    raw = "" if text is None else str(text)
    if mode == "exact":
        return raw
    if mode == "strip":
        return raw.strip()
    if mode == "nfkc_strip":
        return unicodedata.normalize("NFKC", raw).strip()
    return raw.strip()


def _resolve_toy_bench_image_path(image: str, base_dir: Path) -> Path:
    candidate = Path(str(image)).expanduser()
    if candidate.is_absolute():
        return candidate
    by_dataset = base_dir / candidate
    if by_dataset.exists():
        return by_dataset.resolve()
    if candidate.exists():
        return candidate.resolve()
    return by_dataset.resolve()


def _iter_toy_bench_samples(dataset_path: str) -> List[Dict[str, Any]]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    base_dir = path.parent
    samples: List[Dict[str, Any]] = []
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row:
                    continue
                image = (row.get("image") or row.get("path") or row.get("img") or "").strip()
                expected = None
                for key in ("text", "expected", "label"):
                    if key in row:
                        expected = row.get(key)
                        break
                if expected is None:
                    expected = ""
                allowed = row.get("allowed_chars") or row.get("allowed") or row.get("charset")
                sample_id = row.get("id") or row.get("name") or ""
                if not image:
                    continue
                img_path = _resolve_toy_bench_image_path(image, base_dir)
                samples.append(
                    {
                        "id": sample_id.strip() or None,
                        "image": str(img_path),
                        "expected": str(expected),
                        "allowed_chars": str(allowed) if allowed else None,
                    }
                )
        return samples
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            image = obj.get("image") or obj.get("path") or obj.get("img")
            expected = None
            for key in ("text", "expected", "label"):
                if key in obj:
                    expected = obj.get(key)
                    break
            allowed = obj.get("allowed_chars") or obj.get("allowed") or obj.get("charset")
            sample_id = obj.get("id") or obj.get("name") or obj.get("key")
            if not image or expected is None:
                continue
            img_path = _resolve_toy_bench_image_path(str(image), base_dir)
            samples.append(
                {
                    "id": str(sample_id) if sample_id else None,
                    "image": str(img_path),
                    "expected": str(expected),
                    "allowed_chars": str(allowed) if allowed else None,
                    "source_line": lineno,
                }
            )
    return samples


def cli_toy_bench(args) -> None:
    from PIL import Image

    from . import toy_runtime as _toy_runtime

    dataset = getattr(args, "dataset", None) or ""
    dataset = dataset.strip()
    if not dataset:
        print("Missing --dataset")
        return
    compare_mode = getattr(args, "compare", "nfkc_strip") or "nfkc_strip"
    limit = int(getattr(args, "limit", 0) or 0)
    save_fails = bool(getattr(args, "save_fails", False))
    global_allowed = getattr(args, "allowed_chars", None)
    if global_allowed is not None and not str(global_allowed).strip():
        global_allowed = None

    runtime_overrides: Dict[str, Any] = {}
    sweeps = getattr(args, "toy_sweeps", None)
    if sweeps is not None and sweeps > 0:
        runtime_overrides["sweeps"] = sweeps
    force_numeric = getattr(args, "force_numeric", None)
    if force_numeric is not None:
        runtime_overrides["force_numeric"] = force_numeric
    if runtime_overrides:
        configure_toy_runtime(**runtime_overrides)

    samples = _iter_toy_bench_samples(dataset)
    if not samples:
        print("No samples found in", dataset)
        return
    if limit > 0:
        samples = samples[:limit]

    out_dir = Path(args.out).resolve()
    out_results = out_dir / "toy_bench_results.jsonl"
    out_report = out_dir / "toy_bench_report.json"
    fails_dir = out_dir / "toy_bench_fails"
    if save_fails:
        fails_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    correct = 0
    skipped = 0
    conf_sum = 0.0
    conf_ok_sum = 0.0
    conf_bad_sum = 0.0
    ok_count = 0
    bad_count = 0

    with out_results.open("w", encoding="utf-8") as f_out:
        for idx, sample in enumerate(samples, 1):
            image_path = sample.get("image") or ""
            expected = sample.get("expected") or ""
            allowed_chars = global_allowed if global_allowed is not None else sample.get("allowed_chars")
            try:
                img = Image.open(image_path)
            except Exception as exc:
                skipped += 1
                rec = {
                    "id": sample.get("id"),
                    "image": image_path,
                    "expected": expected,
                    "pred": "",
                    "conf": 0.0,
                    "ok": False,
                    "skipped": True,
                    "error": str(exc),
                }
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            pred, conf = _toy_runtime.toy_ocr_text_from_cell(img, allowed_chars=allowed_chars)
            expected_cmp = _normalize_bench_text(expected, compare_mode)
            pred_cmp = _normalize_bench_text(pred, compare_mode)
            ok = pred_cmp == expected_cmp

            total += 1
            conf_sum += float(conf)
            if ok:
                correct += 1
                ok_count += 1
                conf_ok_sum += float(conf)
            else:
                bad_count += 1
                conf_bad_sum += float(conf)

            rec = {
                "id": sample.get("id"),
                "image": image_path,
                "expected": expected,
                "pred": pred,
                "conf": float(conf),
                "ok": bool(ok),
                "compare": compare_mode,
                "allowed_chars": allowed_chars,
            }
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if save_fails and not ok:
                src = Path(image_path)
                base = src.name or f"sample_{idx}.png"
                digest = hashlib.sha1(
                    f"{idx}:{image_path}".encode("utf-8", errors="ignore")
                ).hexdigest()[:10]
                dest = fails_dir / f"{idx:05d}_{digest}_{base}"
                try:
                    if src.exists():
                        dest.write_bytes(src.read_bytes())
                    else:
                        img.save(dest)
                except Exception:
                    pass

    evaluated = total
    accuracy = float(correct) / float(evaluated or 1)
    report = {
        "dataset": str(Path(dataset).resolve()),
        "out_dir": str(out_dir),
        "compare": compare_mode,
        "limit": limit if limit > 0 else None,
        "evaluated": evaluated,
        "correct": correct,
        "accuracy": accuracy,
        "skipped": skipped,
        "avg_conf": float(conf_sum) / float(evaluated or 1),
        "avg_conf_ok": float(conf_ok_sum) / float(ok_count or 1),
        "avg_conf_bad": float(conf_bad_sum) / float(bad_count or 1),
        "results_jsonl": str(out_results),
        "report_json": str(out_report),
        "fails_dir": str(fails_dir) if save_fails else None,
    }
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"Toy bench: {correct}/{evaluated} correct  acc={accuracy:.3f}  avg_conf={report['avg_conf']:.3f}  out={out_dir}"
    )


def _patch_cli_for_export_and_search(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    sub = parser.add_subparsers(dest="cmd")

    def add_common(sp):
        sp.add_argument("--out", type=str, default="out_consensus")
        sp.add_argument("-i", "--input", nargs="*", default=[])
        sp.add_argument(
            "--toy-sweeps",
            type=int,
            default=None,
            help="Clamp toy OCR threshold sweeps (overrides ZOCR_TOY_SWEEPS)",
        )
        group = sp.add_mutually_exclusive_group()
        group.add_argument(
            "--force-numeric",
            dest="force_numeric",
            action="store_true",
            help="Force numeric coercion based on headers",
        )
        group.add_argument(
            "--no-force-numeric",
            dest="force_numeric",
            action="store_false",
            help="Disable numeric coercion",
        )
        sp.set_defaults(force_numeric=None)

    sp = sub.add_parser("export", help="Export JSONL with toy OCR + contextual lines")
    add_common(sp)
    sp.set_defaults(func=cli_export)

    sp = sub.add_parser("index", help="Build local BM25 index from exported JSONL")
    add_common(sp)
    sp.set_defaults(func=cli_index)

    sp = sub.add_parser("stats", help="Inspect cached BM25 metadata")
    sp.add_argument("--json", action="store_true", help="Emit JSON stats instead of text")
    add_common(sp)
    sp.set_defaults(func=cli_stats)

    sp = sub.add_parser("query", help="Query local index (RRF with optional image)")
    sp.add_argument("--query", type=str, default="")
    sp.add_argument("--image-query", type=str, default="")
    sp.add_argument("--topk", type=int, default=10)
    add_common(sp)
    sp.set_defaults(func=cli_query)

    sp = sub.add_parser("toy-bench", help="Evaluate toy OCR against a golden CSV/JSONL dataset")
    sp.add_argument("--dataset", type=str, required=True, help="Path to CSV or JSONL with image/text pairs")
    sp.add_argument(
        "--compare",
        type=str,
        choices=("exact", "strip", "nfkc_strip"),
        default="nfkc_strip",
        help="Text comparison mode (default: nfkc_strip)",
    )
    sp.add_argument("--limit", type=int, default=0, help="Limit evaluated samples (0 = no limit)")
    sp.add_argument("--allowed-chars", type=str, default=None, help="Override allowed character set for all samples")
    sp.add_argument("--save-fails", action="store_true", help="Copy failing crops into out/toy_bench_fails/")
    add_common(sp)
    sp.set_defaults(func=cli_toy_bench)
    return parser


def main():
    p = argparse.ArgumentParser(description="Z‑OCR one‑file (Consensus + MM RAG)")
    p.add_argument("-i", "--input", nargs="*", default=[], help="Images or PDF")
    p.add_argument("--out", default="out_consensus")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--demo", action="store_true")
    p.add_argument(
        "--domain",
        default=None,
        help="Domain keyword profile for the toy OCR lexicon (e.g. invoice, medical_bill)",
    )
    p.add_argument("--bench-iterations", type=int, default=20)
    # CC params
    p.add_argument("--cc-k", type=int, default=31)
    p.add_argument("--cc-c", type=float, default=10.0)
    p.add_argument("--cc-min-area", type=int, default=32)
    p.add_argument("--dp-lambda-factor", type=float, default=2.2)
    p.add_argument("--shape-lambda", type=float, default=4.0)
    p.add_argument("--lambda-alpha", type=float, default=0.7)
    p.add_argument("--iou-thr", type=float, default=0.35)
    p.add_argument("--iou-sigma", type=float, default=0.10)
    p.add_argument("--baseline-segs", type=int, default=4)
    p.add_argument("--baseline-thr-factor", type=float, default=0.7)
    p.add_argument("--baseline-sigma-factor", type=float, default=0.15)
    p.add_argument("--wx", type=int, default=0)
    p.add_argument("--wy", type=int, default=0)
    p.add_argument("--consensus-thr", type=float, default=0.5)
    p.add_argument("--ambiguous-low", type=float, default=0.35)
    p.add_argument("--ambiguous-high", type=float, default=0.65)
    # auto
    p.add_argument(
        "--autocalib",
        nargs="?",
        type=int,
        const=_AUTOCALIB_DEFAULT_SAMPLES,
        default=None,
        metavar="N",
        help=(
            "Auto-calibrate CC params using N sample pages (default %(const)s when "
            "the flag is provided without a value). Pass 0 or omit the flag to disable."
        ),
    )
    p.add_argument(
        "--autotune",
        nargs="?",
        type=int,
        const=_AUTOTUNE_DEFAULT_TRIALS,
        default=None,
        metavar="N",
        help=(
            "Run the unsupervised autotuner for N trials (default %(const)s when the "
            "flag has no explicit value). Pass 0 or omit to skip autotuning."
        ),
    )
    _patch_cli_for_export_and_search(p)
    args = p.parse_args()
    if args.domain is not None:
        dom = args.domain.strip()
        if not dom or dom.lower() in {"", "auto", "default"}:
            os.environ.pop("ZOCR_TESS_DOMAIN", None)
        else:
            os.environ["ZOCR_TESS_DOMAIN"] = dom
    ensure_dir(args.out)
    # subcommands (export/index/query) do not require re-running OCR
    if args.cmd:
        args.func(args)
        return
    if args.demo:
        demo_inputs = _discover_demo_inputs_for_consensus()
        if not demo_inputs:
            p.error(
                "`--demo` requested but no sample inputs were found. "
                "Place demo files under samples/demo_inputs/ or samples/input_demo/."
            )
        pages: List[str] = []
        for it in demo_inputs:
            ext = os.path.splitext(it)[1].lower()
            if ext == ".pdf":
                pages += pdf_to_images_via_poppler(it, dpi=args.dpi)
            else:
                pages.append(it)
        pages = _dedup_input_paths(pages)
        annos = [None] * len(pages)
    else:
        if not args.input:
            p.error("No input. Use --demo or -i.")
        pages = []
        for it in args.input:
            ext = os.path.splitext(it)[1].lower()
            if ext == ".pdf":
                pages += pdf_to_images_via_poppler(it, dpi=args.dpi)
            else:
                pages.append(it)
        pages = _dedup_input_paths(pages)
        annos = [None] * len(pages)
    tab_cfg = {
        "k": args.cc_k,
        "c": args.cc_c,
        "min_area": args.cc_min_area,
        "dp_lambda_factor": args.dp_lambda_factor,
        "shape_lambda": args.shape_lambda,
        "lambda_alpha": args.lambda_alpha,
        "wx": args.wx,
        "wy": args.wy,
        "iou_thr": args.iou_thr,
        "iou_sigma": args.iou_sigma,
        "baseline_segs": args.baseline_segs,
        "baseline_thr_factor": args.baseline_thr_factor,
        "baseline_sigma_factor": args.baseline_sigma_factor,
        "consensus_thr": args.consensus_thr,
        "ambiguous_low": args.ambiguous_low,
        "ambiguous_high": args.ambiguous_high,
    }
    autocalib_samples = _positive_cli_value(args.autocalib)
    autotune_trials = _positive_cli_value(args.autotune)
    if autocalib_samples:
        tab_cfg.update(auto_calibrate_params(pages, autocalib_samples))
    if autotune_trials:
        tab_cfg.update(autotune_params(pages, tab_cfg, trials=autotune_trials))
    cfg = {"table": tab_cfg, "bench_iterations": args.bench_iterations, "eval": True}
    pipe = Pipeline(cfg)
    res, out_json = pipe.run("doc", pages, args.out, annos)
    print("Wrote:", out_json)
    print("Wrote:", os.path.join(args.out, "metrics_by_table.csv"))
    print("Wrote:", os.path.join(args.out, "metrics_aggregate.csv"))


if __name__ == "__main__":
    main()
