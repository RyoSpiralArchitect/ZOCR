# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024 ZOCR contributors

"""Public surface for the consensus OCR stack.

This module houses the lazily-loaded exports so that ``zocr.consensus``'s
``__init__`` can remain a trivial shim, keeping the package bootstrap logic out
of the initializer as requested.
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
        module = import_module(_PUBLIC_MODULES[name], __name__.rsplit(".", 1)[0])
        _LOADED[name] = module
        globals()[name] = module
    return module


def __getattr__(name: str) -> ModuleType:
    return _load_module(name)


def __dir__() -> list[str]:
    return sorted(set(__all__ + list(globals().keys())))

