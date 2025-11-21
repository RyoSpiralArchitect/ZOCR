# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024 ZOCR contributors

"""Pipeline orchestrator wrapper.

This module intentionally avoids importing ``zocr_pipeline`` at module import
time so ``python -m zocr.orchestrator.zocr_pipeline`` can execute without the
runtime warning that occurs when the module is already present in
``sys.modules``.  Callers can still access ``zocr_pipeline`` as an attribute of
this package and it will be imported lazily on first access.
"""

from importlib import import_module
from types import ModuleType
from typing import Any

__all__ = ["zocr_pipeline"]


def __getattr__(name: str) -> Any:
    if name == "zocr_pipeline":
        module: ModuleType = import_module(".zocr_pipeline", __name__)
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__ + list(globals().keys())))
