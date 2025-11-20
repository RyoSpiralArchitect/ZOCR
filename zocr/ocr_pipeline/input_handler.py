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
      attempts to expand pages using ``pdf2image``. The optional
      ``poppler_path`` can be passed to the constructor for environments
      without system-wide Poppler.
    * Otherwise, the file is opened with Pillow and treated as a
      single-page image.
    """

    def __init__(self, poppler_path: str | None = None) -> None:
        self.poppler_path = poppler_path

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
            try:
                from pdf2image import convert_from_path
            except Exception as exc:  # pragma: no cover - import guard
                raise RuntimeError(
                    "pdf2image is required to process PDF documents"
                ) from exc

            images = convert_from_path(path.as_posix(), poppler_path=self.poppler_path)
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

