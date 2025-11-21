from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from zocr.ocr_pipeline import (
    BoundingBox,
    DocumentPipeline,
    RegionType,
    ToyRuntimeTextOCR,
    TwoStageTextOCR,
)
from zocr.ocr_pipeline.cli import build_document_pipeline
from zocr.ocr_pipeline.models import ClassifiedRegion, TextOcrResult


@dataclass
class DummyTextOCR:
    text: str
    confidence: float

    calls: int = 0

    def run(self, region: ClassifiedRegion) -> TextOcrResult:  # type: ignore[override]
        self.calls += 1
        return TextOcrResult(
            region_id=region.region_id,
            text=self.text,
            confidence=self.confidence,
            language="dummy",
        )


def _region() -> ClassifiedRegion:
    image = Image.new("RGB", (10, 10), "white")
    return ClassifiedRegion(
        region_id="r1",
        bounding_box=BoundingBox(x=0, y=0, width=10, height=10),
        classification=RegionType.TEXT,
        confidence=1.0,
        reading_order=0,
        image_crop=image,
    )


def test_two_stage_text_ocr_uses_fallback_on_low_confidence():
    primary = DummyTextOCR(text="", confidence=0.2)
    fallback = DummyTextOCR(text="fallback text", confidence=0.9)
    hybrid = TwoStageTextOCR(primary=primary, fallback=fallback, min_primary_confidence=0.5)

    result = hybrid.run(_region())

    assert result.text == "fallback text"
    assert primary.calls == 1
    assert fallback.calls == 1


def test_two_stage_text_ocr_keeps_primary_when_confident():
    primary = DummyTextOCR(text="primary text", confidence=0.95)
    fallback = DummyTextOCR(text="secondary", confidence=0.4)
    hybrid = TwoStageTextOCR(primary=primary, fallback=fallback, min_primary_confidence=0.5)

    result = hybrid.run(_region())

    assert result.text == "primary text"
    assert primary.calls == 1
    assert fallback.calls == 0


def test_two_stage_text_ocr_retains_primary_when_fallback_empty():
    primary = DummyTextOCR(text="low", confidence=0.2)
    fallback = DummyTextOCR(text="   ", confidence=0.8)
    hybrid = TwoStageTextOCR(primary=primary, fallback=fallback, min_primary_confidence=0.5)

    result = hybrid.run(_region())

    assert result.text == "low"
    assert primary.calls == 1
    assert fallback.calls == 1


def test_toy_runtime_wrapper_invokes_toy_runner(monkeypatch):
    calls = {}

    def fake_runner(image):  # noqa: ANN001
        calls["image"] = image
        return "toy", 0.7

    engine = ToyRuntimeTextOCR(toy_runner=fake_runner)
    result = engine.run(_region())

    assert result.text == "toy"
    assert result.confidence == 0.7
    assert "image" in calls


def test_build_document_pipeline_wires_two_stage_default(monkeypatch):
    # Avoid running the real toy runtime inside this test by patching the class used
    monkeypatch.setattr("zocr.ocr_pipeline.cli.ToyRuntimeTextOCR", lambda: DummyTextOCR(text="p", confidence=1.0))
    monkeypatch.setattr("zocr.ocr_pipeline.cli.TesseractTextOCR", lambda: DummyTextOCR(text="f", confidence=0.5))

    pipeline: DocumentPipeline = build_document_pipeline(use_mocks=False)

    assert isinstance(pipeline.page_pipeline.text_ocr, TwoStageTextOCR)

