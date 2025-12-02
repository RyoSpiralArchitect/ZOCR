"""Public surface for the multi-module core package."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, Dict, Tuple

_ATTR_TO_MODULE_ATTR: Dict[str, Tuple[str, str]] = {
    "augment": (".augmenter", "augment"),
    "build_index": (".indexer", "build_index"),
    "query": (".query_engine", "query"),
    "hybrid_query": (".query_engine", "hybrid_query"),
    "embed_jsonl": (".embedders", "embed_jsonl"),
    "sql_export": (".exporters", "sql_export"),
    "export_rag_bundle": (".exporters", "export_rag_bundle"),
    "monitor": (".monitoring", "monitor"),
    "learn_from_monitor": (".monitoring", "learn_from_monitor"),
    "autotune_unlabeled": (".monitoring", "autotune_unlabeled"),
    "metric_col_over_under_rate": (".monitoring", "metric_col_over_under_rate"),
    "metric_chunk_consistency": (".monitoring", "metric_chunk_consistency"),
    "metric_col_alignment_energy_cached": (
        ".monitoring",
        "metric_col_alignment_energy_cached",
    ),
    "extract_structural_grams": (".structural_grams", "extract_structural_grams"),
    "main": (".zocr_core", "main"),
}

_MODULE_EXPORTS: Dict[str, str] = {
    "augmenter": ".augmenter",
    "base": ".base",
    "domains": ".domains",
    "embedders": ".embedders",
    "exporters": ".exporters",
    "indexer": ".indexer",
    "monitoring": ".monitoring",
    "numba_support": ".numba_support",
    "query_engine": ".query_engine",
    "rendering": ".rendering",
    "tokenization": ".tokenization",
    "structural_grams": ".structural_grams",
}

__all__ = ["zocr_core", *_ATTR_TO_MODULE_ATTR.keys(), *_MODULE_EXPORTS.keys()]

_LOADED: Dict[str, Any] = {}

if TYPE_CHECKING:  # pragma: no cover - import-time type hints only
    from . import (
        augmenter,
        base,
        domains,
        exporters,
        indexer,
        monitoring,
        numba_support,
        query_engine,
        tokenization,
    )
    from .augmenter import augment
    from .indexer import build_index
    from .monitoring import (
        autotune_unlabeled,
        learn_from_monitor,
        metric_chunk_consistency,
        metric_col_alignment_energy_cached,
        metric_col_over_under_rate,
        monitor,
    )
    from .query_engine import hybrid_query, query
    from .exporters import export_rag_bundle, sql_export
    from .zocr_core import main


def _load_module(module_spec: str) -> Any:
    module = _LOADED.get(module_spec)
    if module is None:
        module = import_module(module_spec, __name__.rsplit(".", 1)[0])
        _LOADED[module_spec] = module
    return module


def __getattr__(name: str) -> Any:
    if name == "zocr_core":
        return _load_module(".zocr_core")
    if name in _MODULE_EXPORTS:
        return _load_module(_MODULE_EXPORTS[name])
    try:
        module_spec, attr_name = _ATTR_TO_MODULE_ATTR[name]
    except KeyError as exc:  # pragma: no cover - mirrors AttributeError semantics
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = _load_module(module_spec)
    return getattr(module, attr_name)


def __dir__() -> list[str]:
    return sorted(set(__all__ + list(globals().keys())))
