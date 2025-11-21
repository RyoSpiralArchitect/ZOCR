import pytest

from zocr.ocr_pipeline import BoundingBox, ClassifiedRegion, RegionType, TesseractTextOCR


def test_tesseract_text_ocr_extracts_words(monkeypatch):
    calls = []

    def fake_image_to_data(image, lang, config, output_type):
        calls.append((image, lang, config, output_type))
        return {
            "text": ["Hello", "world", ""],
            "conf": ["95", "85", "-1"],
            "left": [1, 50, 0],
            "top": [2, 52, 0],
            "width": [10, 20, 0],
            "height": [10, 20, 0],
        }

    monkeypatch.setattr(
        "zocr.ocr_pipeline.tesseract.pytesseract.image_to_data",
        fake_image_to_data,
    )

    region = ClassifiedRegion(
        region_id="text-1",
        bounding_box=BoundingBox(x=0, y=0, width=100, height=100),
        classification=RegionType.TEXT,
        confidence=0.9,
        reading_order=0,
        image_crop="image-bytes",
    )

    ocr = TesseractTextOCR(lang="eng")
    result = ocr.run(region)

    assert calls[0][1] == "eng"
    assert "--oem 3" in calls[0][2]
    assert "--psm 6" in calls[0][2]
    assert result.text == "Hello world"
    assert result.confidence == pytest.approx((0.95 + 0.85) / 2)
    assert len(result.words or []) == 2
    assert result.words[0].bounding_box.width == 10


def test_tesseract_text_ocr_requires_image_crop():
    region = ClassifiedRegion(
        region_id="text-1",
        bounding_box=BoundingBox(x=0, y=0, width=10, height=10),
        classification=RegionType.TEXT,
        confidence=0.9,
        reading_order=0,
        image_crop=None,
    )

    ocr = TesseractTextOCR()
    with pytest.raises(ValueError):
        ocr.run(region)


def test_tesseract_text_ocr_respects_env_toggle(monkeypatch):
    monkeypatch.setenv("ZOCR_ALLOW_PYTESSERACT", "0")

    with pytest.raises(RuntimeError):
        TesseractTextOCR()

