#!/usr/bin/env python3
"""Unified CLI entry point for the Z-OCR suite."""
from __future__ import annotations

import importlib
import sys
from textwrap import dedent
from types import ModuleType
from typing import Callable

from ._version import __version__

_COMMAND_TO_MODULE = {
    "pipeline": "zocr.orchestrator.zocr_pipeline",
    "orchestrator": "zocr.orchestrator.zocr_pipeline",
    "run": "zocr.orchestrator.zocr_pipeline",
    "consensus": "zocr.consensus.zocr_consensus",
    "cc": "zocr.consensus.zocr_consensus",
    "core": "zocr.core.zocr_core",
    "simple": "zocr.ocr_pipeline.cli",
    "api": "zocr.api_cli",
    "serve": "zocr.service.cli",
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
          simple              Run the lightweight modular OCR pipeline
          api                 Thin ingestion/query wrapper that prints JSON
          serve               Run the reference FastAPI service
          help                Show this message

        Examples:
          python -m zocr run --input demo --snapshot --seed 12345
          python -m zocr consensus --demo --out out_cc
          python -m zocr core query --jsonl out/doc.mm.jsonl --index out/bm25.pkl --q "total amount"
          python -m zocr serve --host 127.0.0.1 --port 8000
        """
    ).strip()
    print(msg)


def _resolve_entry(module_name: str) -> Callable[[list[str] | None], None]:
    module: ModuleType = importlib.import_module(module_name)
    entry = getattr(module, "main", None)
    if entry is None or not callable(entry):
        raise RuntimeError(f"Module {module_name} does not expose a main() entry point")
    return entry


def _invoke(module_name: str, argv: list[str]) -> None:
    entry = _resolve_entry(module_name)
    old_argv = sys.argv
    try:
        sys.argv = [module_name.rsplit(".", 1)[-1], *argv]
        entry()
    finally:
        sys.argv = old_argv


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        _invoke(_COMMAND_TO_MODULE[_DEFAULT_COMMAND], [])
        return
    cmd = argv[0]
    if cmd in {"-V", "--version", "version"}:
        print(__version__)
        return
    if cmd in {"-h", "--help", "help"}:
        _print_help()
        return
    module = _COMMAND_TO_MODULE.get(cmd)
    if module is None:
        module = _COMMAND_TO_MODULE[_DEFAULT_COMMAND]
        args = argv
    else:
        args = argv[1:]
    _invoke(module, args)


if __name__ == "__main__":  # pragma: no cover
    main()
