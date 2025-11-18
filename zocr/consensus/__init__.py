"""Public surface for the consensus OCR stack.

The package exposes both the heavy runtime/CLI helpers as well as the lighter
building blocks (binarization, morphology, connected components, etc.).
Modules are imported lazily so that orchestrators can probe capabilities
without paying the import cost unless they actually need that component.
"""
from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Dict

__all__ = [
    "binarization",
    "columns",
    "components",
    "morphology",
    "runtime",
    "runtime_primitives",
    "utils",
    "toy_runtime",
    "cli",
    "local_search",
    "zocr_consensus",
]

_PUBLIC_MODULES = {
    "binarization": ".binarization",
    "columns": ".columns",
    "components": ".components",
    "morphology": ".morphology",
    "runtime": ".runtime",
    "runtime_primitives": ".runtime_primitives",
    "utils": ".utils",
    "toy_runtime": ".toy_runtime",
    "cli": ".cli",
    "local_search": ".local_search",
    "zocr_consensus": ".zocr_consensus",
}

_LOADED: Dict[str, ModuleType] = {}


def _load_module(name: str) -> ModuleType:
    if name not in _PUBLIC_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = _LOADED.get(name)
    if module is None:
        module = import_module(_PUBLIC_MODULES[name], __name__)
        _LOADED[name] = module
        globals()[name] = module
    return module


def __getattr__(name: str) -> ModuleType:
    return _load_module(name)


def __dir__() -> list[str]:
    return sorted(set(__all__ + list(globals().keys())))
