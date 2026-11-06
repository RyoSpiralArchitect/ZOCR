import math
import os
import platform

import pytest

from zocr.consensus.ocr_quality import assess_recognition
from zocr.consensus.vision import validate_response, VisionRecognizer


def test_high_confidence_cannot_override_an_unsupported_language():
    result = assess_recognition("mxaxW+", .99, requested_language="ja", supported_languages=["en"])
    assert result["status"] == "abstained"
    assert result["text"] == ""
    assert result["raw_text"] == "mxaxW+"
    assert "unsupported_language" in result["reasons"]


def test_low_confidence_preserves_the_raw_hypothesis_without_accepting_it():
    result = assess_recognition("14", .4, requested_language="ja", supported_languages=["ja", "en"])
    assert result["text"] == ""
    assert result["raw_text"] == "14"
    assert result["status"] == "abstained"
    assert result["review_required"] is True
    assert "low_confidence" in result["reasons"]


def test_high_confidence_is_still_not_human_verification():
    result = assess_recognition("デザイナー", .99, requested_language="ja", supported_languages=["ja"])
    assert result["text"] == "デザイナー"
    assert result["status"] == "recognized"
    assert result["review_required"] is True
    assert result["confidence_calibration"] == "unverified"


@pytest.mark.parametrize("confidence", [math.nan, math.inf, True, -1, 1.1])
def test_malformed_confidence_is_rejected(confidence):
    with pytest.raises(ValueError):
        assess_recognition("14", confidence, requested_language="ja", supported_languages=["ja"])


@pytest.mark.parametrize("engine", ["not-an-engine", "tesseract:totally-missing", "toy:ja"])
def test_explicit_ocr_requests_never_silently_fall_back_to_toy(monkeypatch, engine):
    from zocr.consensus import toy_runtime

    monkeypatch.setattr(toy_runtime, "_OCR_BACKEND_CACHE", {})
    monkeypatch.setattr(toy_runtime, "pytesseract", None)
    with pytest.raises((ValueError, RuntimeError)):
        toy_runtime._resolve_ocr_backend(engine)


def test_vision_coordinates_convert_from_bottom_left_to_source_pixels():
    result = validate_response({"engine": "apple_vision", "revision": 3, "observations": [
        {"text": "デザイナー", "confidence": .99, "bbox": [.1, .2, .3, .4]},
    ]}, 1000, 500)
    assert result["lines"][0]["bbox"] == [100, 199, 400, 400] or result["lines"][0]["bbox"] == [100, 200, 400, 400]
    assert result["model_sha256"] is None
    assert result["language_correction"] is False


def test_vision_clips_subpixel_overshoot_and_retains_the_reported_geometry():
    box = [-.0006925, .21, .94, .57]
    result = validate_response({"engine": "apple_vision", "revision": 3, "observations": [
        {"text": "ポジション", "confidence": 1., "bbox": box},
    ]}, 400, 100)
    line, = result["lines"]
    assert line["bbox"][0] == 0
    assert line["bbox_clipped"] is True
    assert line["normalized_bbox"] == box


@pytest.mark.parametrize("box", [[-.1, 0, .2, .2], [0, 0, 2, .2], [0, 0, 0, .2], [0, 0, math.nan, .2]])
def test_vision_rejects_invalid_coordinates(box):
    with pytest.raises(ValueError):
        validate_response({"engine": "apple_vision", "revision": 3, "observations": [
            {"text": "14", "confidence": .99, "bbox": box},
        ]}, 100, 100)


@pytest.mark.skipif(platform.system() != "Darwin" or os.environ.get("ZOCR_TEST_VISION") != "1",
                    reason="Set ZOCR_TEST_VISION=1 on macOS with Swift for real native OCR")
def test_real_apple_vision_recognizes_pixels_without_external_services():
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (180, 90), "white")
    ImageDraw.Draw(image).text((15, 15), "14", font=ImageFont.load_default(size=40), fill="black")
    with VisionRecognizer() as recognizer:
        result = recognizer.recognize(image)
    assert "14" in "".join(line["text"] for line in result["lines"])
    assert result["languages"] == ["ja-JP", "en-US"]
    assert result["build"]["binary_sha256"]
