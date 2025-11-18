"""Optional dependency diagnostics for the ZOCR stack."""
from __future__ import annotations

import shutil
from typing import Any, Dict, Optional

__all__ = ["collect_dependency_diagnostics"]


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return default


def _describe_pdf(consensus_module: Optional[Any]) -> Dict[str, Any]:
    diag: Dict[str, Any] = {}
    detect_pdf_backends = None
    if consensus_module is not None:
        detect_pdf_backends = _safe_getattr(consensus_module, "detect_pdf_raster_backends")
    if callable(detect_pdf_backends):
        try:
            diag["pdf_raster"] = detect_pdf_backends()
        except Exception:
            pass
    if "pdf_raster" not in diag:
        poppler_path = shutil.which("pdftoppm")
        diag["pdf_raster"] = {
            "status": "ready" if poppler_path else "missing",
            "active": "poppler_pdftoppm" if poppler_path else None,
            "poppler_pdftoppm": {
                "status": "available" if poppler_path else "missing",
                "path": poppler_path,
            },
            "hint": None
            if poppler_path
            else "Install poppler-utils (pdftoppm) or `pip install pypdfium2` for multi-page PDF rasterisation",
        }
    pdf_diag = diag.get("pdf_raster")
    if isinstance(pdf_diag, dict):
        for backend in ("poppler_pdftoppm", "pypdfium2"):
            entry = pdf_diag.get(backend)
            if isinstance(entry, dict):
                diag[backend] = dict(entry)
    return diag


def _describe_core_acceleration(core_module: Optional[Any]) -> Dict[str, Any]:
    diag: Dict[str, Any] = {}
    numba_enabled = bool(_safe_getattr(core_module, "_HAS_NUMBA", False)) if core_module else False
    numba_parallel = bool(_safe_getattr(core_module, "_HAS_NUMBA_PARALLEL", False)) if core_module else False
    if numba_enabled:
        detail = "Numba acceleration active"
        if not numba_parallel:
            detail += " (atomic.add unavailable; DF reduction running serially)"
    else:
        detail = "Falling back to pure Python BM25 scoring"
    diag["numba"] = {
        "status": "enabled" if numba_enabled else "python-fallback",
        "detail": detail,
    }
    libc_path = _safe_getattr(core_module, "_LIBC_PATH") if core_module else None
    diag["c_extensions"] = {
        "status": "loaded" if libc_path else "python-fallback",
        "path": libc_path,
        "detail": "Custom SIMD/Thomas/rle helpers" if libc_path else "Using pure Python/NumPy helpers",
    }
    return diag


def _describe_numpy(numpy_module: Optional[Any]) -> Dict[str, Any]:
    np_mod = numpy_module
    if np_mod is None:
        try:
            import numpy as _np  # type: ignore
        except Exception:
            _np = None  # type: ignore
        np_mod = _np
    version = None
    if np_mod is not None:
        version = _safe_getattr(np_mod, "__version__")
    return {
        "numpy": {
            "status": "available" if np_mod is not None else "missing",
            "version": version,
        }
    }


def _describe_pillow() -> Dict[str, Any]:
    try:
        import PIL  # type: ignore

        pillow_version = _safe_getattr(PIL, "__version__")
    except Exception:
        pillow_version = None
    return {
        "pillow": {
            "status": "available" if pillow_version else "unknown",
            "version": pillow_version,
        }
    }


def collect_dependency_diagnostics(
    *,
    consensus_module: Optional[Any] = None,
    core_module: Optional[Any] = None,
    numpy_module: Optional[Any] = None,
) -> Dict[str, Any]:
    """Summarise optional dependency health for operators."""

    diag: Dict[str, Any] = {}
    diag.update(_describe_pdf(consensus_module))
    diag.update(_describe_core_acceleration(core_module))
    diag.update(_describe_numpy(numpy_module))
    diag.update(_describe_pillow())
    return diag
