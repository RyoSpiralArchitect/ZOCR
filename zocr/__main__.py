#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified CLI entry point for the Z-OCR suite."""
from __future__ import annotations

import runpy
import sys
from textwrap import dedent

_COMMAND_TO_MODULE = {
    "pipeline": "zocr.orchestrator.zocr_pipeline",
    "orchestrator": "zocr.orchestrator.zocr_pipeline",
    "run": "zocr.orchestrator.zocr_pipeline",
    "consensus": "zocr.consensus.zocr_consensus",
    "cc": "zocr.consensus.zocr_consensus",
    "core": "zocr.core.zocr_core",
}

_DEFAULT_COMMAND = "pipeline"


def _print_help() -> None:
    msg = dedent(
        """
        Usage:
          python -m zocr [command] [args...]

        Commands:
          pipeline | run      Execute the end-to-end orchestrator (default)
          consensus | cc      Access the consensus/table reconstruction CLI
          core                Access the multi-domain core CLI (augment/index/query/...)
          help                Show this message

        Examples:
          python -m zocr run --input demo --snapshot --seed 12345
          python -m zocr consensus --demo --out out_cc
          python -m zocr core query --jsonl out/doc.mm.jsonl --index out/bm25.pkl --q "total amount"
        """
    ).strip()
    print(msg)


def _run_module(module: str, argv: list[str]) -> None:
    old_argv = sys.argv
    try:
        sys.argv = [module, *argv]
        runpy.run_module(module, run_name="__main__")
    finally:
        sys.argv = old_argv


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        _run_module(_COMMAND_TO_MODULE[_DEFAULT_COMMAND], [])
        return
    cmd = argv[0]
    if cmd in {"-h", "--help", "help"}:
        _print_help()
        return
    module = _COMMAND_TO_MODULE.get(cmd)
    if module is None:
        module = _COMMAND_TO_MODULE[_DEFAULT_COMMAND]
        args = argv
    else:
        args = argv[1:]
    _run_module(module, args)


if __name__ == "__main__":
    main()
