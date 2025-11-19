"""Public surface for the multi-module core package."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict

__all__ = [
    "zocr_core",
    "augment",
    "build_index",
    "query",
    "sql_export",
    "export_rag_bundle",
    "monitor",
    "learn_from_monitor",
    "autotune_unlabeled",
    "metric_col_over_under_rate",
    "metric_chunk_consistency",
    "metric_col_alignment_energy_cached",
    "main",
]

_PROXIED_ATTRS = set(__all__) - {"zocr_core"}
_LOADED: Dict[str, Any] = {}


def _load_core() -> Any:
    module = _LOADED.get("core")
    if module is None:
        module = import_module(".zocr_core", __name__.rsplit(".", 1)[0])
        _LOADED["core"] = module
    return module


def __getattr__(name: str) -> Any:
    module = _load_core()
    if name == "zocr_core":
        return module
    if name in _PROXIED_ATTRS:
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__ + list(globals().keys())))
