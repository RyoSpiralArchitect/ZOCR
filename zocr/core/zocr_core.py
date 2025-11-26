"""Compatibility shim exposing the multi-module core API."""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional

__all__ = [
    "augment",
    "build_index",
    "query",
    "embed_jsonl",
    "sql_export",
    "export_rag_bundle",
    "monitor",
    "learn_from_monitor",
    "autotune_unlabeled",
    "metric_col_over_under_rate",
    "metric_chunk_consistency",
    "metric_col_alignment_energy_cached",
    "extract_structural_grams",
    "main",
]


def augment(*args, **kwargs):
    from .augmenter import augment as _impl

    return _impl(*args, **kwargs)


def build_index(*args, **kwargs):
    from .indexer import build_index as _impl

    return _impl(*args, **kwargs)


def query(*args, **kwargs):
    from .query_engine import query as _impl

    return _impl(*args, **kwargs)


def embed_jsonl(*args, **kwargs):
    from .embedders import embed_jsonl as _impl

    return _impl(*args, **kwargs)


def sql_export(*args, **kwargs):
    from .exporters import sql_export as _impl

    return _impl(*args, **kwargs)


def export_rag_bundle(*args, **kwargs):
    from .exporters import export_rag_bundle as _impl

    return _impl(*args, **kwargs)


def extract_structural_grams(*args, **kwargs):
    from .structural_grams import extract_structural_grams as _impl

    return _impl(*args, **kwargs)


def metric_col_over_under_rate(*args, **kwargs):
    from .monitoring import metric_col_over_under_rate as _impl

    return _impl(*args, **kwargs)


def metric_chunk_consistency(*args, **kwargs):
    from .monitoring import metric_chunk_consistency as _impl

    return _impl(*args, **kwargs)


def metric_col_alignment_energy_cached(*args, **kwargs):
    from .monitoring import metric_col_alignment_energy_cached as _impl

    return _impl(*args, **kwargs)


def monitor(*args, **kwargs):
    from .monitoring import monitor as _impl

    return _impl(*args, **kwargs)


def learn_from_monitor(*args, **kwargs):
    from .monitoring import learn_from_monitor as _impl

    return _impl(*args, **kwargs)


def autotune_unlabeled(*args, **kwargs):
    from .monitoring import autotune_unlabeled as _impl

    return _impl(*args, **kwargs)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser("ZOCR Multi-domain Core")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("augment")
    sp.add_argument("--jsonl", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--lambda-shape", type=float, default=4.5)
    sp.add_argument("--lambda-refheight", type=int, default=1000)
    sp.add_argument("--lambda-alpha", type=float, default=0.7)
    sp.add_argument("--org-dict", default=None)

    sp = sub.add_parser("index")
    sp.add_argument("--jsonl", required=True)
    sp.add_argument("--index", required=True)

    sp = sub.add_parser("query")
    sp.add_argument("--jsonl", required=True)
    sp.add_argument("--index", required=True)
    sp.add_argument("--q", default="")
    sp.add_argument("--image", default=None)
    sp.add_argument("--topk", type=int, default=10)
    sp.add_argument("--w-bm25", type=float, default=1.0)
    sp.add_argument("--w-kw", type=float, default=0.6)
    sp.add_argument("--w-img", type=float, default=0.3)
    sp.add_argument("--w-sym", type=float, default=0.45)
    sp.add_argument("--domain", default="invoice")

    sp = sub.add_parser("embed")
    sp.add_argument("--jsonl", required=True)
    sp.add_argument("--out", default=None)
    sp.add_argument("--model", default=os.environ.get("ZOCR_EMBED_MODEL"))
    sp.add_argument("--provider", default=os.environ.get("ZOCR_EMBED_PROVIDER", "sentence-transformers"))
    sp.add_argument("--text-field", default="text")
    sp.add_argument("--batch-size", type=int, default=32)
    sp.add_argument("--no-normalize", action="store_true")
    sp.add_argument("--aws-profile", default=os.environ.get("AWS_PROFILE"))
    sp.add_argument("--aws-region", default=os.environ.get("AWS_REGION"))
    sp.add_argument("--aws-endpoint-url", default=os.environ.get("AWS_ENDPOINT_URL"))

    sp = sub.add_parser("sql")
    sp.add_argument("--jsonl", required=True)
    sp.add_argument("--outdir", required=True)
    sp.add_argument("--prefix", default="invoice")

    sp = sub.add_parser("structural-grams")
    sp.add_argument("--jsonl", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--max-ngram", type=int, default=4)

    sp = sub.add_parser("rag")
    sp.add_argument("--jsonl", required=True)
    sp.add_argument("--outdir", required=True)
    sp.add_argument("--domain", default=None)
    sp.add_argument("--limit", type=int, default=40)

    sp = sub.add_parser("monitor")
    sp.add_argument("--jsonl", required=True)
    sp.add_argument("--index", required=True)
    sp.add_argument("--k", type=int, default=10)
    sp.add_argument("--views-log", default=None)
    sp.add_argument("--gt-jsonl", default=None)
    sp.add_argument("--out", required=True)
    sp.add_argument("--domain", default="invoice")

    sp = sub.add_parser("autotune")
    sp.add_argument("--jsonl", required=True)
    sp.add_argument("--index", required=True)
    sp.add_argument("--outdir", required=True)
    sp.add_argument("--budget", type=int, default=30)
    sp.add_argument("--domain", default=None)

    sp = sub.add_parser("learn-monitor")
    sp.add_argument("--monitor-csv", required=True)
    sp.add_argument("--profile-in", required=True)
    sp.add_argument("--profile-out", default=None)
    sp.add_argument("--domain", default=None)

    args = parser.parse_args(argv)
    if args.cmd == "augment":
        n = augment(
            args.jsonl,
            args.out,
            args.lambda_shape,
            args.lambda_refheight,
            args.lambda_alpha,
            args.org_dict,
        )
        print(f"Augmented {n} records -> {args.out}")
        return
    if args.cmd == "index":
        build_index(args.jsonl, args.index)
        print(f"Indexed -> {args.index}")
        return
    if args.cmd == "query":
        res = query(
            args.index,
            args.jsonl,
            args.q,
            args.image,
            args.topk,
            args.w_bm25,
            args.w_kw,
            args.w_img,
            args.w_sym,
            args.domain,
        )
        for i, (score, ob) in enumerate(res, 1):
            filters = (ob.get("meta") or {}).get("filters", {})
            scores = (ob.get("meta") or {}).get("retrieval_scores", {})
            print(
                f"{i:2d}. {score:.3f} page={ob.get('page')} r={ob.get('row')} c={ob.get('col')} "
                f"sym={scores.get('symbolic', 0):.2f} amt={filters.get('amount')} date={filters.get('date')} "
                f"text='{(ob.get('text') or '')[:60]}'"
            )
        return
    if args.cmd == "embed":
        if not args.model:
            raise SystemExit("Please pass --model or set ZOCR_EMBED_MODEL")
        out_path = embed_jsonl(
            args.jsonl,
            args.out,
            args.model,
            text_field=args.text_field,
            batch_size=args.batch_size,
            normalize=not args.no_normalize,
            provider=args.provider,
            aws_profile=args.aws_profile,
            aws_region=args.aws_region,
            aws_endpoint_url=args.aws_endpoint_url,
        )
        print(out_path)
        return
    if args.cmd == "sql":
        paths = sql_export(args.jsonl, args.outdir, args.prefix)
        print(json.dumps(paths, ensure_ascii=False, indent=2))
        return
    if args.cmd == "structural-grams":
        n = extract_structural_grams(args.jsonl, args.out, max_ngram=args.max_ngram)
        print(f"Extracted {n} grams -> {args.out}")
        return
    if args.cmd == "rag":
        bundle = export_rag_bundle(args.jsonl, args.outdir, args.domain, limit_per_section=args.limit)
        print(json.dumps(bundle, ensure_ascii=False, indent=2))
        return
    if args.cmd == "monitor":
        row = monitor(
            args.jsonl,
            args.index,
            args.k,
            args.out,
            views_log=args.views_log,
            gt_jsonl=args.gt_jsonl,
            domain=args.domain,
        )
        print(json.dumps(row, ensure_ascii=False, indent=2))
        return
    if args.cmd == "autotune":
        result = autotune_unlabeled(
            args.jsonl,
            args.index,
            args.outdir,
            budget=args.budget,
            domain_hint=args.domain,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.cmd == "learn-monitor":
        result = learn_from_monitor(
            args.monitor_csv,
            args.profile_in,
            profile_json_out=args.profile_out,
            domain_hint=args.domain,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    parser.print_help()


if __name__ == "__main__":  # pragma: no cover
    main()
