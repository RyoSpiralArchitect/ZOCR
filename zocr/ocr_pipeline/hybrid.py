"""Hybrid text OCR utilities combining toy and Tesseract engines.

The toy runtime provides a lightweight, low-latency recogniser that works well
on the synthetic/demo fonts used throughout Z-OCR. Tesseract is more accurate
on difficult inputs (handwriting-like glyphs, skew, noisy scans) but slower. A
two-stage wrapper lets the toy engine answer quickly while still falling back
to Tesseract when confidence is low or text is missing, reinforcing the
"ground truth before LLM" contract emphasized in Phase 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

from PIL import Image

from .interfaces import TextOCR
from .models import ClassifiedRegion, RegionType, TextOcrResult


class ToyRuntimeTextOCR(TextOCR):
    """Wrap the consensus toy runtime for use inside the modular pipeline."""

    def __init__(self, toy_runner: Callable[[Image.Image], Tuple[str, float]] | None = None) -> None:
        if toy_runner is None:
            try:  # Lazy import to keep upstream dependencies optional
                from zocr.consensus.toy_runtime import toy_ocr_text_from_cell
            except Exception as exc:  # pragma: no cover - guarded for environments without consensus deps
                raise ImportError("Toy runtime is unavailable") from exc

            toy_runner = toy_ocr_text_from_cell

        self._toy_runner = toy_runner

    def run(self, region: ClassifiedRegion) -> TextOcrResult:
        if region.classification != RegionType.TEXT:
            raise ValueError("ToyRuntimeTextOCR can only process text regions")
        if not isinstance(region.image_crop, Image.Image):
            raise ValueError("ToyRuntimeTextOCR requires a PIL.Image crop on the region")

        text, confidence = self._toy_runner(region.image_crop)
        return TextOcrResult(
            region_id=region.region_id,
            text=text,
            confidence=float(confidence or 0.0),
            language="toy",
            engine="toy_runtime",
        )


@dataclass
class TwoStageTextOCR(TextOCR):
    """Run a fast primary OCR engine with an accuracy-first fallback."""

    primary: TextOCR
    fallback: TextOCR
    min_primary_confidence: float = 0.5
    min_primary_chars: int = 1

    def run(self, region: ClassifiedRegion) -> TextOcrResult:
        primary_result = self.primary.run(region)

        if not self._needs_fallback(primary_result):
            return primary_result

        fallback_result = self.fallback.run(region)
        if self._accept_fallback(fallback_result, primary_result):
            return fallback_result
        return primary_result

    def _needs_fallback(self, result: TextOcrResult) -> bool:
        if result.confidence < self.min_primary_confidence:
            return True
        return len(result.text.strip()) < self.min_primary_chars

    @staticmethod
    def _accept_fallback(candidate: TextOcrResult, baseline: TextOcrResult) -> bool:
        has_text = bool(candidate.text.strip())
        if not has_text:
            return False
        if candidate.confidence > baseline.confidence:
            return True
        if not baseline.text.strip():
            return True
        return False

