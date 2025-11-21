# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024 ZOCR contributors

"""CLI exposure for the thin ingestion/query surface.

This module keeps behaviour minimal and predictable so it can be called from
shell scripts or wired behind an HTTP handler. Subcommands mirror the
``ingest_job`` and ``query_job`` helpers and print JSON to stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from zocr.api import IngestRequest, ingest_job, query_job


class _JsonAction(argparse.Action):
    """Parse simple key=value pairs into a dictionary.

    This keeps the CLI friendly while allowing callers to forward selected
    ``zocr_pipeline`` kwargs without committing to a fixed schema.
    """

    def __call__(self, parser, namespace, values, option_string=None):  # type: ignore[override]
        parsed: Dict[str, Any] = {}
        for value in values:
            if "=" not in value:
                raise argparse.ArgumentError(self, "Expected KEY=VALUE pairs")
            key, raw = value.split("=", 1)
            try:
                parsed[key] = json.loads(raw)
            except json.JSONDecodeError:
                parsed[key] = raw
        setattr(namespace, self.dest, parsed)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="zocr api", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Run ingestion and print artifact locations")
    ingest.add_argument("files", nargs="+", help="Input files or URIs")
    ingest.add_argument("--tenant", required=True, dest="tenant_id")
    ingest.add_argument("--job-id", dest="job_id")
    ingest.add_argument("--out-root", default="episodes", dest="out_root")
    ingest.add_argument("--domain-hint")
    ingest.add_argument("--resume", action="store_true")
    ingest.add_argument("--dry-run", action="store_true")
    ingest.add_argument(
        "--pipeline-kw",
        nargs="*",
        action=_JsonAction,
        default={},
        metavar="KEY=JSON",
        help="Optional zocr_pipeline kwargs (parsed as JSON when possible)",
    )

    query = sub.add_parser("query", help="Query existing artifacts and print response")
    query.add_argument("--tenant", required=True, dest="tenant_id")
    query.add_argument("--job-id", required=True, dest="job_id")
    query.add_argument("--base-dir", default="episodes")
    query.add_argument("--conversation-id")
    query.add_argument("--query", default="")
    query.add_argument("--mode", default="analysis")

    return parser


def _print_json(payload: Dict[str, Any]) -> None:
    json.dump(payload, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")


def _handle_ingest(args: argparse.Namespace) -> None:
    req = IngestRequest(
        tenant_id=args.tenant_id,
        files=args.files,
        domain_hint=args.domain_hint,
        job_id=args.job_id,
        out_root=args.out_root,
        resume=args.resume,
        dry_run=args.dry_run,
        pipeline_kwargs=args.pipeline_kw,
    )
    result = ingest_job(req)
    _print_json(asdict(result))


def _handle_query(args: argparse.Namespace) -> None:
    result = query_job(
        job_id=args.job_id,
        tenant_id=args.tenant_id,
        base_dir=args.base_dir,
        conversation_id=args.conversation_id,
        query=args.query,
        mode=args.mode,
    )
    payload: Dict[str, Any] = asdict(result)
    _print_json(payload)


def main(argv: Optional[List[str]] = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "ingest":
        _handle_ingest(args)
    elif args.command == "query":
        _handle_query(args)
    else:  # pragma: no cover - argparse enforces choices
        parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":  # pragma: no cover
    main()
