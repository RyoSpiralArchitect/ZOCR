# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024 ZOCR contributors

"""Public surface and helpers for the Z-OCR suite."""

from __future__ import annotations

from importlib import import_module
import sys
from types import ModuleType
from typing import Any, Dict, Iterable, Optional

__all__ = [
    "zocr_consensus",
    "zocr_core",
    "zocr_pipeline",
    "describe_stack",
    "inspect_stack",
]

_ATTR_TO_SPEC: Dict[str, str] = {
    "zocr_consensus": ".consensus.zocr_consensus",
    "zocr_core": ".core.zocr_core",
    "zocr_pipeline": ".orchestrator.zocr_pipeline",
}

_ATTR_TO_ALIAS: Dict[str, str] = {
    "zocr_consensus": "zocr_onefile_consensus",
    "zocr_core": "zocr_multidomain_core",
    "zocr_pipeline": "zocr_pipeline_allinone",
}

_LOADED: Dict[str, Any] = {}


def _load(name: str) -> Any:
    if name not in _ATTR_TO_SPEC:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = _LOADED.get(name)
    if module is None:
        module = import_module(_ATTR_TO_SPEC[name], __name__.rsplit(".", 1)[0])
        alias = _ATTR_TO_ALIAS.get(name)
        if alias:
            sys.modules.setdefault(alias, module)
        _LOADED[name] = module
    return module


def _optional_import(path: str) -> Optional[ModuleType]:
    try:
        return import_module(path)
    except Exception:  # pragma: no cover - optional dependency import
        return None


def describe_stack(*, include_exports: bool = False) -> Dict[str, Any]:
    """Return a structured summary of the consensus/core/pipeline stack."""

    inspector = import_module(".orchestrator.stack_inspector", __name__.rsplit(".", 1)[0])
    pipeline_module = _optional_import("zocr.orchestrator.zocr_pipeline")
    consensus_module = _optional_import("zocr.consensus")
    core_module = _optional_import("zocr.core.zocr_core")
    return inspector.describe_full_stack(
        pipeline_module=pipeline_module,
        consensus_module=consensus_module,
        core_module=core_module,
        include_exports=include_exports,
    )


def inspect_stack(
    *,
    include_exports: bool = False,
    sections: Optional[Iterable[str]] = None,
) -> str:
    """Render a human-readable report of the Z-OCR runtime stack."""

    inspector = import_module(".orchestrator.stack_inspector", __name__.rsplit(".", 1)[0])
    summary = describe_stack(include_exports=include_exports)
    return inspector.format_stack_report(
        summary,
        include_exports=include_exports,
        sections=sections,
    )


def __getattr__(name: str) -> Any:
    return _load(name)


def __dir__() -> list[str]:
    return sorted(set(__all__ + list(globals().keys())))
