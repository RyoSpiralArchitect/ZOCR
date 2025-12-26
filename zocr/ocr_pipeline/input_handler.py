"""Input handling utilities for preparing pages for the OCR pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import List

from PIL import Image

from .interfaces import InputHandler
from .models import DocumentInput, PageInput


class BasicInputHandler(InputHandler):
    """Simple input handler that supports images and PDFs.

    * If ``images`` are provided on the :class:`DocumentInput`, each is
      emitted as a :class:`PageInput` in order.
    * If a ``file_path`` is provided and ends with ``.pdf``, the handler
      first attempts to expand pages using ``pdf2image`` (Poppler). If that
      dependency is missing or fails, it falls back to a pure-Python
      ``PyMuPDF`` renderer so environments without system packages remain
      usable. The optional ``poppler_path`` can be passed to the constructor
      for environments without system-wide Poppler.
    * Otherwise, the file is opened with Pillow and treated as a
      single-page image.
    """

    def __init__(self, poppler_path: str | None = None, default_pdf_dpi: int = 200) -> None:
        self.poppler_path = poppler_path
        self.default_pdf_dpi = default_pdf_dpi

    def _render_pdf_with_pdf2image(self, path: Path, dpi: int | None) -> List[Image.Image]:
        from pdf2image import convert_from_path

        kwargs = {}
        if self.poppler_path:
            kwargs["poppler_path"] = self.poppler_path
        kwargs["dpi"] = dpi or self.default_pdf_dpi
        return convert_from_path(path.as_posix(), **kwargs)

    def _render_pdf_with_pymupdf(self, path: Path, dpi: int | None) -> List[Image.Image]:
        import importlib

        spec = importlib.util.find_spec("fitz")
        if spec is None:  # pragma: no cover - optional dependency
            raise RuntimeError("PyMuPDF (fitz) is required to process PDF documents")

        fitz = importlib.import_module("fitz")
        scale = (dpi or self.default_pdf_dpi) / 72.0
        matrix = fitz.Matrix(scale, scale)
        images: List[Image.Image] = []
        with fitz.open(path.as_posix()) as doc:  # type: ignore[attr-defined]
            for page_index in range(doc.page_count):
                page = doc.load_page(page_index)
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                mode = "RGB"
                if pix.n == 1:
                    mode = "L"
                elif pix.n >= 4:
                    mode = "RGBA"
                images.append(Image.frombytes(mode, (pix.width, pix.height), pix.samples))
        return images

    def _expand_pdf(self, path: Path, dpi: int | None) -> List[Image.Image]:
        errors: list[str] = []
        try:
            return self._render_pdf_with_pdf2image(path, dpi)
        except Exception as exc:
            errors.append(f"pdf2image: {exc}")

        try:
            return self._render_pdf_with_pymupdf(path, dpi)
        except Exception as exc:
            errors.append(f"pymupdf: {exc}")
            raise RuntimeError(
                "Install pdf2image with Poppler or PyMuPDF (fitz) to process PDF documents; "
                f"attempts failed ({'; '.join(errors)})"
            ) from exc

    def load(self, document: DocumentInput) -> List[PageInput]:
        if document.images:
            return [
                PageInput(
                    document_id=document.document_id,
                    page_number=idx + 1,
                    image=img,
                    dpi=document.dpi,
                )
                for idx, img in enumerate(document.images)
            ]

        if not document.file_path:
            raise ValueError("DocumentInput requires either images or a file_path")

        path = Path(document.file_path)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            images = self._expand_pdf(path, document.dpi)
        else:
            images = [Image.open(path.as_posix())]

        return [
            PageInput(
                document_id=document.document_id,
                page_number=idx + 1,
                image=img,
                dpi=document.dpi,
            )
            for idx, img in enumerate(images)
        ]
