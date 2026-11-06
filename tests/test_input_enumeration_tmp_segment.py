from __future__ import annotations

from pathlib import Path

from PIL import Image

from zocr.orchestrator import zocr_pipeline


def test_collect_pages_honors_explicit_file_under_tmp_segment(tmp_path: Path) -> None:
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    img_path = tmp_dir / "page.png"
    Image.new("RGB", (16, 16), (255, 255, 255)).save(img_path)

    pages = zocr_pipeline._collect_pages([str(img_path)], dpi=200)
    assert pages == [str(img_path)]

    # Directory enumeration should still skip `tmp` segments by default.
    assert zocr_pipeline._collect_pages([str(tmp_dir)], dpi=200) == []


def test_explicit_output_image_is_allowed_without_recursing_into_outputs(tmp_path):
    output = tmp_path / "out-real-probe"
    output.mkdir()
    image = output / "page.png"
    Image.new("RGB", (16, 16), "white").save(image)
    assert zocr_pipeline._collect_pages([str(image)], dpi=200) == [str(image)]
    assert zocr_pipeline._collect_pages([str(output)], dpi=200) == []

