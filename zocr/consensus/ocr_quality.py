from __future__ import annotations

import math
import unicodedata


def assess_recognition(text: str, confidence: float, *, requested_language: str,
                       supported_languages, minimum_confidence: float = .8) -> dict:
    if (not isinstance(text, str) or isinstance(confidence, bool)
            or not isinstance(confidence, (int, float)) or not math.isfinite(confidence)
            or not 0 <= confidence <= 1 or not math.isfinite(minimum_confidence)
            or not 0 <= minimum_confidence <= 1):
        raise ValueError("Invalid OCR observation or confidence threshold")
    if not isinstance(requested_language, str) or not requested_language:
        raise ValueError("An explicit OCR language is required")
    language = requested_language.lower().split("-")[0]
    supported = {value.lower().split("-")[0] for value in supported_languages}
    reasons = []
    if language not in supported:
        reasons.append("unsupported_language")
    if not text.strip():
        reasons.append("empty_ocr_output")
    if any(char == "\ufffd" or unicodedata.category(char) in {"Co", "Cs", "Cc"}
           for char in text if char not in "\t\n\r"):
        reasons.append("unreadable_text")
    if confidence < minimum_confidence:
        reasons.append("low_confidence")
    return {"text": "" if reasons else text, "raw_text": text, "confidence": float(confidence),
            "status": "abstained" if reasons else "recognized", "reasons": reasons,
            "review_required": True, "confidence_calibration": "unverified",
            "policy": "Heuristic abstention, not a calibrated correctness probability or human approval."}
