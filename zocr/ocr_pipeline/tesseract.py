"""Text OCR implementation backed by pytesseract.

This module provides a production-ready implementation of the ``TextOCR``
protocol using pytesseract. It emphasizes accuracy by enabling LSTM-based OCR
(``--oem 3``) and a layout-aware page segmentation mode (``--psm 6``) by
default. Word-level metadata is preserved so downstream components can reason
about confidence and positioning.
"""
from __future__ import annotations

import os
from statistics import mean
from typing import List

import pytesseract
from pytesseract import Output

from .interfaces import TextOCR
from .models import BoundingBox, ClassifiedRegion, RegionType, TextOcrResult, WordInfo


def _pytesseract_allowed() -> bool:
    raw = os.environ.get("ZOCR_ALLOW_PYTESSERACT")
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


class TesseractTextOCR(TextOCR):
    """Run OCR on text regions using pytesseract.

    Args:
        lang: Language hint passed to Tesseract (e.g., ``"eng"`` or
            ``"eng+deu"``).
        oem: OCR Engine Mode. The default ``3`` enables Tesseract's LSTM engine
            for higher accuracy.
        psm: Page segmentation mode. The default ``6`` treats the input as a
            uniform block of text for balanced accuracy and speed.
        extra_config: Additional custom flags forwarded to pytesseract.
    """

    def __init__(self, lang: str = "eng", oem: int = 3, psm: int = 6, extra_config: str = "") -> None:
        if not _pytesseract_allowed():
            raise RuntimeError(
                "pytesseract is disabled by ZOCR_ALLOW_PYTESSERACT; set it to 1/true to enable"
            )
        self.lang = lang
        base_config = f"--oem {oem} --psm {psm}"
        self.config = f"{base_config} {extra_config}".strip()

    def run(self, region: ClassifiedRegion) -> TextOcrResult:
        if region.classification != RegionType.TEXT:
            raise ValueError("TesseractTextOCR can only process text regions")
        if region.image_crop is None:
            raise ValueError("ClassifiedRegion.image_crop is required for OCR")

        data = pytesseract.image_to_data(
            region.image_crop,
            lang=self.lang,
            config=self.config,
            output_type=Output.DICT,
        )

        words: List[WordInfo] = []
        confidences: List[float] = []

        for text, conf_str, left, top, width, height in zip(
            data.get("text", []),
            data.get("conf", []),
            data.get("left", []),
            data.get("top", []),
            data.get("width", []),
            data.get("height", []),
        ):
            if not text or conf_str is None or conf_str == "-1":
                continue

            conf = float(conf_str) / 100.0
            word_box = BoundingBox(x=int(left), y=int(top), width=int(width), height=int(height))
            words.append(WordInfo(word=text, bounding_box=word_box, confidence=conf))
            confidences.append(conf)

        text_content = " ".join(word.word for word in words)
        overall_conf = mean(confidences) if confidences else 0.0

        return TextOcrResult(
            region_id=region.region_id,
            text=text_content,
            confidence=overall_conf,
            language=self.lang,
            words=words if words else None,
        )

