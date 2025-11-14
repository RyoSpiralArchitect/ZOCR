"""Z-OCR public package surface."""

from __future__ import annotations

from importlib import import_module as _import_module
import sys as _sys
from typing import Any, Dict

__all__ = [
    "zocr_consensus",
    "zocr_core",
    "zocr_pipeline",
]

# Mapping of public attribute -> import path
_ATTR_TO_SPEC: Dict[str, str] = {
    "zocr_consensus": ".consensus.zocr_consensus",
    "zocr_core": ".core.zocr_core",
    "zocr_pipeline": ".orchestrator.zocr_pipeline",
}

# Optional aliases that legacy import paths still expect to resolve.
_ATTR_TO_ALIAS: Dict[str, str] = {
    "zocr_consensus": "zocr_onefile_consensus",
    "zocr_core": "zocr_multidomain_core",
    "zocr_pipeline": "zocr_pipeline_allinone",
}

_loaded: Dict[str, Any] = {}


def _load(name: str) -> Any:
    if name not in _ATTR_TO_SPEC:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = _loaded.get(name)
    if module is None:
        module = _import_module(_ATTR_TO_SPEC[name], __name__)
        alias = _ATTR_TO_ALIAS.get(name)
        if alias:
            _sys.modules.setdefault(alias, module)
        _loaded[name] = module
    return module


def __getattr__(name: str) -> Any:
    return _load(name)


def __dir__() -> list[str]:
    return sorted(set(__all__ + list(globals().keys())))
