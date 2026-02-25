"""Utilities for inspecting the orchestration/core/consensus stack."""
from __future__ import annotations

from importlib import import_module
from pathlib import Path
import re
import sys
from types import ModuleType
from typing import Any, Dict, Iterable, List, Optional

_CONSENSUS_ROLES = {
    "binarization": "Image binarization primitives",
    "columns": "Column clustering heuristics",
    "components": "Connected component helpers",
    "morphology": "Morphological transforms",
    "runtime": "Full consensus runtime entrypoints",
    "runtime_primitives": "Shared execution-core helpers",
    "utils": "Utility helpers shared across consensus modules",
    "toy_runtime": "Toy OCR runtime scaffolding",
    "cli": "Standalone CLI entrypoints",
    "local_search": "Local BM25/img search helpers",
    "zocr_consensus": "Compatibility shim exporting runtime + CLI",
}

_CORE_MODULES = [
    "augmenter",
    "base",
    "domains",
    "exporters",
    "indexer",
    "monitoring",
    "numba_support",
    "query_engine",
    "tokenization",
    "zocr_core",
]

_CORE_ROLES = {
    "augmenter": "Data augmentation helpers feeding the core",
    "base": "Shared base classes and interfaces",
    "domains": "Domain-specific heuristics and keyword tables",
    "exporters": "Core export sinks (SQL, JSONL, etc.)",
    "indexer": "Search index construction",
    "monitoring": "Monitoring/metrics entrypoints",
    "numba_support": "Optional numba acceleration helpers",
    "query_engine": "RAG/query orchestration",
    "tokenization": "Tokenizer helpers for downstream tasks",
    "zocr_core": "Multi-domain pipeline facade",
}


def _first_line(text: Optional[str]) -> str:
    if not text:
        return ""
    for line in text.strip().splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _module_summary(module: ModuleType, name: str, *, role: Optional[str], include_exports: bool) -> Dict[str, Any]:
    exports = getattr(module, "__all__", None)
    export_list: Optional[List[str]] = None
    if include_exports and isinstance(exports, Iterable):
        export_list = sorted(str(item) for item in exports)
    summary: Dict[str, Any] = {
        "name": name,
        "module": module.__name__,
        "path": getattr(module, "__file__", None),
        "doc": _first_line(getattr(module, "__doc__", "")),
        "role": role,
    }
    if export_list is not None:
        summary["exports"] = export_list
    return summary


def _module_error_summary(
    name: str,
    module_path: str,
    *,
    role: Optional[str],
    error: Exception,
) -> Dict[str, Any]:
    return {
        "name": name,
        "module": module_path,
        "role": role,
        "error": str(error),
    }


def describe_consensus_stack(*, include_exports: bool = False) -> Dict[str, Any]:
    pkg = import_module("zocr.consensus")
    modules: List[Dict[str, Any]] = []
    for name in getattr(pkg, "__all__", []):
        module = None
        load_error: Optional[Exception] = None
        try:
            module = getattr(pkg, name)
        except Exception:  # pragma: no cover - depends on optional deps
            try:
                module = import_module(f"{pkg.__name__}.{name}")
            except Exception as inner:
                load_error = inner
        if module is None:
            if load_error is None:
                load_error = RuntimeError(f"failed to load {pkg.__name__}.{name}")
            modules.append(
                _module_error_summary(
                    name,
                    f"{pkg.__name__}.{name}",
                    role=_CONSENSUS_ROLES.get(name),
                    error=load_error,
                )
            )
            continue
        modules.append(
            _module_summary(
                module,
                name,
                role=_CONSENSUS_ROLES.get(name),
                include_exports=include_exports,
            )
        )
    alias_target = sys.modules.get("zocr_onefile_consensus")
    alias_info = {
        "name": "zocr_onefile_consensus",
        "module": getattr(alias_target, "__name__", None) if alias_target else None,
        "loaded": bool(alias_target),
    }
    return {
        "package": pkg.__name__,
        "module_count": len(modules),
        "modules": modules,
        "aliases": [alias_info],
    }


def describe_core_stack(*, include_exports: bool = False) -> Dict[str, Any]:
    base_pkg = "zocr.core"
    modules: List[Dict[str, Any]] = []
    for name in _CORE_MODULES:
        try:
            module = import_module(f"{base_pkg}.{name}")
        except Exception as exc:
            modules.append(
                _module_error_summary(
                    name,
                    f"{base_pkg}.{name}",
                    role=_CORE_ROLES.get(name),
                    error=exc,
                )
            )
            continue
        modules.append(
            _module_summary(
                module,
                name,
                role=_CORE_ROLES.get(name),
                include_exports=include_exports,
            )
        )
    alias_target = sys.modules.get("zocr_multidomain_core")
    alias_info = {
        "name": "zocr_multidomain_core",
        "module": getattr(alias_target, "__name__", None) if alias_target else None,
        "loaded": bool(alias_target),
    }
    return {
        "package": base_pkg,
        "module_count": len(modules),
        "modules": modules,
        "aliases": [alias_info],
    }


def _extract_touchpoints(source: str, symbol: str) -> List[str]:
    direct = re.findall(rf"{re.escape(symbol)}\.([A-Za-z0-9_]+)", source)
    getattr_calls = re.findall(
        rf"getattr\({re.escape(symbol)},\s*(['\"])([A-Za-z0-9_]+)\1",
        source,
    )
    names = sorted({*direct, *(name for _, name in getattr_calls)})
    return names


def _touchpoint_status(names: List[str], module: Optional[ModuleType]) -> Dict[str, Any]:
    details: List[Dict[str, Any]] = []
    missing: List[str] = []
    for name in names:
        available = hasattr(module, name) if module is not None else None
        details.append({"name": name, "available": available})
        if available is False:
            missing.append(name)
    return {
        "count": len(names),
        "attributes": details,
        "missing": missing,
    }


def _summarize_plugins(pipeline_module: Optional[ModuleType]) -> List[Dict[str, Any]]:
    if pipeline_module is None:
        return []
    registry = getattr(pipeline_module, "PLUGINS", None)
    if not isinstance(registry, dict):
        return []
    summary: List[Dict[str, Any]] = []
    for stage in sorted(registry.keys()):
        funcs = registry.get(stage) or []
        summary.append({"stage": stage, "count": len(funcs)})
    return summary


def describe_pipeline_touchpoints(
    *,
    pipeline_module: Optional[ModuleType],
    consensus_module: Optional[ModuleType],
    core_module: Optional[ModuleType],
    include_plugins: bool = True,
) -> Dict[str, Any]:
    pipeline_path = Path(__file__).with_name("zocr_pipeline.py")
    try:
        source = pipeline_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        source = ""
    consensus_names = _extract_touchpoints(source, "zocr_onefile_consensus")
    core_names = _extract_touchpoints(source, "zocr_multidomain_core")
    summary: Dict[str, Any] = {
        "path": str(pipeline_path),
        "consensus": _touchpoint_status(consensus_names, consensus_module),
        "core": _touchpoint_status(core_names, core_module),
    }
    if include_plugins:
        summary["plugins"] = _summarize_plugins(pipeline_module)
    return summary


def describe_full_stack(
    *,
    pipeline_module: Optional[ModuleType] = None,
    consensus_module: Optional[ModuleType] = None,
    core_module: Optional[ModuleType] = None,
    include_exports: bool = False,
) -> Dict[str, Any]:
    summary = {
        "consensus": describe_consensus_stack(include_exports=include_exports),
        "core": describe_core_stack(include_exports=include_exports),
    }
    summary["pipeline"] = describe_pipeline_touchpoints(
        pipeline_module=pipeline_module,
        consensus_module=consensus_module,
        core_module=core_module,
    )
    return summary


def format_stack_report(
    summary: Dict[str, Any],
    *,
    include_exports: bool = False,
    sections: Optional[Iterable[str]] = None,
) -> str:
    selected = set(sections or ("consensus", "core", "pipeline"))
    lines: List[str] = []
    if "consensus" in selected:
        consensus = summary.get("consensus") or {}
        lines.append("[Consensus stack]")
        lines.append(f"Modules: {consensus.get('module_count', 0)}")
        for module in consensus.get("modules", []):
            role = module.get("role") or module.get("doc") or ""
            lines.append(f" - {module.get('name')}: {role}")
            if include_exports and module.get("exports"):
                exports = ", ".join(module.get("exports") or [])
                lines.append(f"    exports: {exports}")
        aliases = consensus.get("aliases") or []
        for alias in aliases:
            lines.append(
                f"   alias {alias.get('name')}: {'loaded' if alias.get('loaded') else 'not loaded'}"
            )
        if "core" in selected or "pipeline" in selected:
            lines.append("")

    if "core" in selected:
        core = summary.get("core") or {}
        lines.append("[Core stack]")
        lines.append(f"Modules: {core.get('module_count', 0)}")
        for module in core.get("modules", []):
            role = module.get("role") or module.get("doc") or ""
            lines.append(f" - {module.get('name')}: {role}")
            if include_exports and module.get("exports"):
                exports = ", ".join(module.get("exports") or [])
                lines.append(f"    exports: {exports}")
        aliases = core.get("aliases") or []
        for alias in aliases:
            lines.append(
                f"   alias {alias.get('name')}: {'loaded' if alias.get('loaded') else 'not loaded'}"
            )
        if "pipeline" in selected:
            lines.append("")

    if "pipeline" in selected:
        pipeline = summary.get("pipeline") or {}
        lines.append("[Pipeline integration]")
        path = pipeline.get("path")
        if path:
            lines.append(f"Source: {path}")
        for label in ("consensus", "core"):
            section = pipeline.get(label) or {}
            names = section.get("attributes") or []
            missing = section.get("missing") or []
            lines.append(
                f" - {label} touchpoints: {section.get('count', len(names))} (missing: {', '.join(missing) if missing else 'none'})"
            )
        plugins = pipeline.get("plugins") or []
        if plugins:
            lines.append(" - Registered plugin stages:")
            for plug in plugins:
                lines.append(f"    * {plug.get('stage')}: {plug.get('count')} function(s)")
        else:
            lines.append(" - Registered plugin stages: none")
    return "\n".join(line for line in lines if line is not None)


__all__ = [
    "describe_consensus_stack",
    "describe_core_stack",
    "describe_pipeline_touchpoints",
    "describe_full_stack",
    "format_stack_report",
]
