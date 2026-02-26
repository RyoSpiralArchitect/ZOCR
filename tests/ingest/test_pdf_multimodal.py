import json
from pathlib import Path

import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

pytest.importorskip("fitz")
pytest.importorskip("pandas")

from zocr.ingest.pdf_multimodal import (
    chunk_page_graph,
    extract_pdf_multimodal,
    save_page_graph_json,
)


def _make_sample_pdf(path: Path) -> None:
    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter

    # Page 1: intro text, raster figure with caption, and a two-column paragraph
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, height - 72, "Multimodal PDF Sample")

    # Vector figure rectangle
    c.rect(72, height - 240, 220, 130, stroke=1, fill=0)
    c.setFont("Helvetica", 10)
    c.drawString(72, height - 270, "Figure 1: System overview diagram")

    # Two-column body text
    c.setFont("Helvetica", 11)
    left_text = "Left column narrative discussing the architecture and its flow. " * 2
    right_text = "Right column notes explaining metrics and deployment details. " * 2
    c.drawString(72, height - 320, left_text)
    c.drawString(width / 2 + 20, height - 320, right_text)
    c.showPage()

    # Page 2: table with caption and first half of cross-page vector diagram
    c.setFont("Helvetica", 10)
    c.drawString(72, height - 72, "Table 1: Metrics summary")
    # Simple table grid
    top = height - 100
    left = 72
    cell_w = 100
    cell_h = 24
    rows, cols = 3, 3
    for r in range(rows + 1):
        y = top - r * cell_h
        c.line(left, y, left + cols * cell_w, y)
    for col in range(cols + 1):
        x = left + col * cell_w
        c.line(x, top, x, top - rows * cell_h)

    # Cross-page vector diagram part 1 near page bottom
    c.setFont("Helvetica", 10)
    c.drawString(72, 130, "Figure 2: Cross-page system flow part 1")
    c.rect(72, 80, 180, 40, stroke=1, fill=0)
    c.showPage()

    # Page 3: continuation of cross-page diagram
    c.setFont("Helvetica", 10)
    c.drawString(72, height - 140, "Figure 2: Cross-page system flow part 2")
    c.rect(72, height - 200, 180, 40, stroke=1, fill=0)
    c.save()


@pytest.fixture(scope="module")
def sample_pdf_path(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("pdfs") / "technical_sample.pdf"
    _make_sample_pdf(path)
    return path


@pytest.fixture(scope="module")
def page_graph(sample_pdf_path):
    return extract_pdf_multimodal(str(sample_pdf_path))


def test_extract_counts_and_coordinates(page_graph, sample_pdf_path):
    assert len(page_graph.pages) == 3

    text_count = sum(1 for p in page_graph.pages for o in p["objects"] if o.type == "text")
    figure_count = sum(1 for p in page_graph.pages for o in p["objects"] if "figure" in o.type)
    table_count = sum(1 for p in page_graph.pages for o in p["objects"] if o.type == "table")

    assert text_count >= 6
    assert figure_count >= 2
    assert table_count == 1

    for page in page_graph.pages:
        width, height = page["width"], page["height"]
        for obj in page["objects"]:
            x0, y0, x1, y1 = obj.bbox
            assert 0 <= x0 < x1 <= width
            assert 0 <= y0 < y1 <= height
            if obj.caption_bbox:
                cx0, cy0, cx1, cy1 = obj.caption_bbox
                assert 0 <= cx0 < cx1 <= width
                assert 0 <= cy0 < cy1 <= height


def test_chunking_groups_visual_regions(page_graph, tmp_path):
    chunks = chunk_page_graph(page_graph)

    figure_chunk = next(c for c in chunks if "System overview diagram" in c["content"])
    table_chunk = next(c for c in chunks if "Metrics summary" in c["content"])
    stitched = next(c for c in chunks if "Cross-page system flow" in c["content"] and len(c["pages"]) > 1)

    assert figure_chunk["pages"] == [1]
    assert set(stitched["pages"]) == {2, 3}
    assert table_chunk["type"] == "table"

    # Persist JSON to ensure serialization remains stable
    out_json = tmp_path / "graph.json"
    save_page_graph_json(page_graph, out_json)
    saved = json.loads(out_json.read_text())
    assert saved["pages"][0]["objects"][0]["bbox"]
