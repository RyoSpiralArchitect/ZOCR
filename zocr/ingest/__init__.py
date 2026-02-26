"""PDF and document ingestion helpers.

The multimodal PDF ingestion stack depends on optional packages (e.g.
pdfminer/PyMuPDF). To keep the base install lightweight, we expose these symbols
via lazy attribute access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "PageGraph",
    "PageObject",
    "chunk_page_graph",
    "extract_pdf_multimodal",
    "save_page_graph_json",
    "save_page_graph_parquet",
]

if TYPE_CHECKING:  # pragma: no cover
    from .pdf_multimodal import (  # noqa: F401
        PageGraph,
        PageObject,
        chunk_page_graph,
        extract_pdf_multimodal,
        save_page_graph_json,
        save_page_graph_parquet,
    )


def __getattr__(name: str) -> Any:  # pragma: no cover - import-time indirection
    if name not in __all__:
        raise AttributeError(name)
    try:
        from . import pdf_multimodal as _mod
    except Exception as exc:
        raise ImportError(
            "zocr.ingest requires optional dependencies. Install with "
            "`pip install 'zocr-suite[multimodal]'`."
        ) from exc
    return getattr(_mod, name)


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(list(globals().keys()) + __all__)
