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
    compare_primary_engines: Tuple[str, ...] = ()
    min_fallback_confidence: float = 0.35
    fallback_confidence_margin: float = 0.05

    def run(self, region: ClassifiedRegion) -> TextOcrResult:
        primary_result = self.primary.run(region)

        compare_with_fallback = self._should_compare(primary_result)
        if not self._needs_fallback(primary_result) and not compare_with_fallback:
            return primary_result

        fallback_result = self.fallback.run(region)
        if self._accept_fallback(
            fallback_result,
            primary_result,
            prefer_when_usable=compare_with_fallback,
        ):
            return fallback_result
        return primary_result

    def _needs_fallback(self, result: TextOcrResult) -> bool:
        if result.confidence < self.min_primary_confidence:
            return True
        return len(result.text.strip()) < self.min_primary_chars

    def _should_compare(self, result: TextOcrResult) -> bool:
        if not self.compare_primary_engines:
            return False
        engine = (result.engine or "").lower()
        return engine in {item.lower() for item in self.compare_primary_engines}

    def _accept_fallback(
        self,
        candidate: TextOcrResult,
        baseline: TextOcrResult,
        *,
        prefer_when_usable: bool = False,
    ) -> bool:
        has_text = bool(candidate.text.strip())
        if not has_text:
            return False
        if candidate.confidence > baseline.confidence + self.fallback_confidence_margin:
            return True
        if not baseline.text.strip():
            return True
        if prefer_when_usable and candidate.confidence >= self.min_fallback_confidence:
            candidate_text = candidate.text.strip()
            baseline_text = baseline.text.strip()
            if (
                candidate.confidence + self.fallback_confidence_margin >= baseline.confidence
                and len(candidate_text) >= len(baseline_text)
            ):
                return True
            if len(candidate_text) >= max(len(baseline_text) + 3, int(len(baseline_text) * 1.5)):
                return True
        return False
