"""Pytest configuration shared across the suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw


@pytest.fixture
def table_image(tmp_path):
    image = Image.new("RGB", (240, 120), "white")
    draw = ImageDraw.Draw(image)
    for y in (5, 40, 75, 110):
        draw.line((5, y, 235, y), fill="black", width=2)
    for x in (5, 120, 235):
        draw.line((x, 5, x, 110), fill="black", width=2)
    for x, y, text in ((15, 15, "Item"), (135, 15, "Qty"),
                       (15, 50, "A"), (135, 50, "1"),
                       (15, 85, "B"), (135, 85, "2")):
        draw.text((x, y), text, fill="black")
    source = tmp_path / "page.png"
    image.save(source)
    return source


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
