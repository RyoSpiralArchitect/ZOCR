"""Multimodal PDF ingestion utilities.

This module extracts text spans, tables, raster images, and vector diagrams
with bounding boxes from PDF files. It persists the result as a page graph
(JSON/Parquet) and supports chunking into retrieval-ready regions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import base64
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import fitz  # PyMuPDF
import pandas as pd
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTFigure, LTLine, LTRect

BBox = Tuple[float, float, float, float]


@dataclass
class PageObject:
    """Representation of an extracted object on a page."""

    id: str
    type: str
    bbox: BBox
    page: int
    text: str | None = None
    caption: str | None = None
    caption_bbox: Optional[BBox] = None
    thumbnail: Optional[str] = None
    references: List[str] = field(default_factory=list)


@dataclass
class PageGraph:
    """Container for all objects extracted from a PDF."""

    pages: List[Dict[str, Any]]
    meta: Dict[str, Any] = field(default_factory=dict)


CaptionPrefixes = ("figure", "fig.", "table", "diagram")


def _rect_to_bbox(rect: Sequence[float]) -> BBox:
    return float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3])


def _clamp_bbox(bbox: BBox, width: float, height: float) -> BBox:
    return (
        max(0.0, min(bbox[0], width)),
        max(0.0, min(bbox[1], height)),
        max(0.0, min(bbox[2], width)),
        max(0.0, min(bbox[3], height)),
    )


def _bbox_union(a: BBox, b: BBox) -> BBox:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def _overlap_ratio(a: BBox, b: BBox) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area = (a[2] - a[0]) * (a[3] - a[1])
    return inter / area if area else 0.0


def _horizontal_overlap(a: BBox, b: BBox) -> float:
    x0 = max(a[0], b[0])
    x1 = min(a[2], b[2])
    width = a[2] - a[0]
    return max(0.0, x1 - x0) / width if width else 0.0


def _encode_thumbnail(page: fitz.Page, bbox: BBox, scale: float = 1.0) -> str:
    clip = fitz.Rect(bbox)
    pix = page.get_pixmap(clip=clip, matrix=fitz.Matrix(scale, scale))
    return base64.b64encode(pix.tobytes("png")).decode("ascii")


def _collect_pdfminer_shapes(pdf_path: str) -> Dict[int, List[BBox]]:
    """Collect line/rect boxes per page using pdfminer for table detection."""
    shapes: Dict[int, List[BBox]] = {}
    for page_no, layout in enumerate(extract_pages(pdf_path), start=1):
        boxes: List[BBox] = []
        for element in layout:
            if isinstance(element, (LTLine, LTRect)):
                boxes.append(_rect_to_bbox(element.bbox))
            elif isinstance(element, LTFigure):
                # descend to capture nested lines within figures
                for child in element:
                    if isinstance(child, (LTLine, LTRect)):
                        boxes.append(_rect_to_bbox(child.bbox))
        shapes[page_no] = boxes
    return shapes


def _cluster_boxes(boxes: List[BBox], margin: float = 5.0, min_count: int = 6) -> List[BBox]:
    padded: List[BBox] = [
        (
            b[0] - margin / 2,
            b[1] - margin / 2,
            b[2] + margin / 2,
            b[3] + margin / 2,
        )
        for b in boxes
    ]

    visited = [False] * len(padded)
    components: List[List[int]] = []

    def intersects(a: BBox, b: BBox) -> bool:
        return not (b[2] < a[0] or b[0] > a[2] or b[3] < a[1] or b[1] > a[3])

    for idx in range(len(padded)):
        if visited[idx]:
            continue
        stack = [idx]
        component = []
        while stack:
            i = stack.pop()
            if visited[i]:
                continue
            visited[i] = True
            component.append(i)
            for j in range(len(padded)):
                if visited[j]:
                    continue
                if intersects(padded[i], padded[j]):
                    stack.append(j)
        components.append(component)

    clusters: List[BBox] = []
    for comp in components:
        if len(comp) < min_count:
            continue
        xs0, ys0, xs1, ys1 = zip(*(padded[i] for i in comp))
        clusters.append((min(xs0), min(ys0), max(xs1), max(ys1)))

    return [c for c in clusters if (c[2] - c[0]) > 20 and (c[3] - c[1]) > 10]


def _is_caption_text(text: str) -> bool:
    lowered = text.strip().lower()
    return any(lowered.startswith(prefix) for prefix in CaptionPrefixes)


def _assign_captions(page_objects: List[PageObject]) -> None:
    figures = [o for o in page_objects if o.type in {"figure", "vector_figure"}]
    tables = [o for o in page_objects if o.type == "table"]
    texts = [o for o in page_objects if o.type == "text"]
    for text_obj in texts:
        if not text_obj.text:
            continue
        content = text_obj.text.strip()
        if not _is_caption_text(content):
            continue
        target_pool = figures if content.lower().startswith(("figure", "fig.")) or "diagram" in content.lower() else tables
        best: Optional[PageObject] = None
        best_distance = float("inf")
        for candidate in target_pool:
            horiz_overlap = _horizontal_overlap(candidate.bbox, text_obj.bbox)
            distance = text_obj.bbox[1] - candidate.bbox[3]
            distance = distance if distance >= 0 else abs(distance)
            if horiz_overlap < 0.2:
                continue
            if distance < best_distance:
                best_distance = distance
                best = candidate
        if best:
            best.caption = content
            best.caption_bbox = text_obj.bbox
            best.references.append(text_obj.id)
            text_obj.references.append(best.id)


def _attach_references(page_objects: List[PageObject]) -> None:
    figures = [o for o in page_objects if o.type in {"figure", "vector_figure"}]
    tables = [o for o in page_objects if o.type == "table"]
    for obj in page_objects:
        if obj.type != "text" or not obj.text:
            continue
        text_lower = obj.text.lower()
        for fig in figures:
            if fig.caption and fig.caption.lower().split(":")[0] in text_lower:
                obj.references.append(fig.id)
        for tbl in tables:
            if tbl.caption and tbl.caption.lower().split(":")[0] in text_lower:
                obj.references.append(tbl.id)


def extract_pdf_multimodal(pdf_path: str, thumbnail_scale: float = 0.5) -> PageGraph:
    """Extract text, tables, figures, and diagrams from a PDF into a page graph."""

    doc = fitz.open(pdf_path)
    shape_boxes = _collect_pdfminer_shapes(pdf_path)
    pages: List[Dict[str, Any]] = []

    for page_index, page in enumerate(doc):
        page_number = page_index + 1
        width, height = page.rect.width, page.rect.height
        objects: List[PageObject] = []

        # Text blocks
        for i, block in enumerate(page.get_text("blocks")):
            if len(block) < 5:
                continue
            bbox = _clamp_bbox(_rect_to_bbox(block[0:4]), width, height)
            text = block[4].strip()
            objects.append(
                PageObject(id=f"text-{page_number}-{i}", type="text", bbox=bbox, page=page_number, text=text)
            )

        # Raster images from text raw dict
        raw = page.get_text("rawdict")
        for block in raw.get("blocks", []):
            if block.get("type") == 1 and "bbox" in block:
                bbox = _clamp_bbox(_rect_to_bbox(block["bbox"]), width, height)
                thumb = _encode_thumbnail(page, bbox, scale=thumbnail_scale)
                objects.append(
                    PageObject(
                        id=f"figure-{page_number}-{block.get('number', len(objects))}",
                        type="figure",
                        bbox=bbox,
                        page=page_number,
                        thumbnail=thumb,
                    )
                )

        # Tables via clustered lines
        boxes = shape_boxes.get(page_number, [])
        table_boxes = list(_cluster_boxes(boxes))
        for k, table_bbox in enumerate(table_boxes):
            table_bbox = _clamp_bbox(table_bbox, width, height)
            thumb = _encode_thumbnail(page, table_bbox, scale=thumbnail_scale)
            objects.append(
                PageObject(
                    id=f"table-{page_number}-{k}",
                    type="table",
                    bbox=table_bbox,
                    page=page_number,
                    thumbnail=thumb,
                )
            )

        # Vector drawings that are not part of table grids
        for j, drawing in enumerate(page.get_drawings()):
            if not drawing.get("rect"):
                continue
            bbox = _clamp_bbox(_rect_to_bbox(drawing["rect"]), width, height)
            width_span = bbox[2] - bbox[0]
            height_span = bbox[3] - bbox[1]
            if width_span <= 1 or height_span <= 1:
                continue
            if any(
                bbox[0] >= tb[0] - 1 and bbox[1] >= tb[1] - 1 and bbox[2] <= tb[2] + 1 and bbox[3] <= tb[3] + 1
                for tb in table_boxes
            ):
                continue
            if table_boxes:
                if (width_span < 15 or height_span < 15) and any(
                    _horizontal_overlap(tb, bbox) > 0.5 or _horizontal_overlap(bbox, tb) > 0.5 for tb in table_boxes
                ):
                    continue
            objects.append(
                PageObject(
                    id=f"vector-{page_number}-{j}",
                    type="vector_figure",
                    bbox=bbox,
                    page=page_number,
                )
            )

        _assign_captions(objects)
        _attach_references(objects)

        pages.append({"page": page_number, "width": width, "height": height, "objects": objects})

    return PageGraph(pages=pages, meta={"source": pdf_path})


def save_page_graph_json(graph: PageGraph, path: str) -> None:
    serializable = {
        "meta": graph.meta,
        "pages": [
            {
                "page": page["page"],
                "width": page["width"],
                "height": page["height"],
                "objects": [o.__dict__ for o in page["objects"]],
            }
            for page in graph.pages
        ],
    }
    with open(path, "w", encoding="utf-8") as fw:
        json.dump(serializable, fw, indent=2)


def save_page_graph_parquet(graph: PageGraph, path: str) -> None:
    records: List[Dict[str, Any]] = []
    for page in graph.pages:
        for obj in page["objects"]:
            record = {
                "page": page["page"],
                "width": page["width"],
                "height": page["height"],
                "id": obj.id,
                "type": obj.type,
                "bbox": obj.bbox,
                "text": obj.text,
                "caption": obj.caption,
                "caption_bbox": obj.caption_bbox,
                "references": obj.references,
            }
            records.append(record)
    df = pd.DataFrame.from_records(records)
    df.to_parquet(path, index=False)


def _merge_cross_page_figures(chunks: List[Dict[str, Any]], page_heights: Dict[int, float]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    chunks_sorted = sorted(chunks, key=lambda c: (min(c["pages"]), c["bbox"][1]))
    for chunk in chunks_sorted:
        if merged:
            last = merged[-1]
            same_type = last["type"] == chunk["type"]
            consecutive = max(last["pages"]) + 1 == min(chunk["pages"])
            horiz_overlap = _horizontal_overlap(last["bbox"], chunk["bbox"])
            near_edge = page_heights[last["pages"][-1]] - last["bbox"][3] < 120 or chunk["bbox"][1] < 120
            part_hint = "part" in (last.get("content") or "").lower() or "part" in (chunk.get("content") or "").lower()
            if same_type and consecutive and horiz_overlap > 0.3 and (near_edge or part_hint):
                last["pages"] = sorted(set(last["pages"] + chunk["pages"]))
                last["bbox"] = _bbox_union(last["bbox"], chunk["bbox"])
                last["content"] = (last.get("content") or "") + " " + (chunk.get("content") or "")
                last["source_objects"] += chunk["source_objects"]
                continue
        merged.append(chunk)
    return merged


def chunk_page_graph(graph: PageGraph) -> List[Dict[str, Any]]:
    """Build retrieval chunks from visual regions."""

    chunks: List[Dict[str, Any]] = []
    page_heights = {p["page"]: p["height"] for p in graph.pages}

    for page in graph.pages:
        width = page["width"]
        text_blocks = [o for o in page["objects"] if o.type == "text"]
        figures = [o for o in page["objects"] if o.type in {"figure", "vector_figure"}]
        tables = [o for o in page["objects"] if o.type == "table"]

        # figure/table chunks with captions
        for obj in figures + tables:
            caption_text = obj.caption or ""
            content = caption_text or (obj.text or obj.type)
            bbox = obj.bbox
            if obj.caption_bbox:
                bbox = _bbox_union(bbox, obj.caption_bbox)
            chunks.append(
                {
                    "id": f"chunk-{obj.id}",
                    "type": obj.type,
                    "pages": [obj.page],
                    "bbox": bbox,
                    "content": content,
                    "source_objects": [obj.id],
                    "references": obj.references,
                }
            )

        # multi-column text regions
        if text_blocks:
            midpoint = width / 2
            columns = {"left": [], "right": []}
            for tb in text_blocks:
                col = "left" if tb.bbox[0] <= midpoint else "right"
                columns[col].append(tb)
            for col_name, blocks in columns.items():
                if not blocks:
                    continue
                blocks_sorted = sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))
                text_content = " ".join(b.text or "" for b in blocks_sorted)
                bbox = blocks_sorted[0].bbox
                for b in blocks_sorted[1:]:
                    bbox = _bbox_union(bbox, b.bbox)
                chunks.append(
                    {
                        "id": f"chunk-page{page['page']}-{col_name}",
                        "type": "text_region",
                        "pages": [page["page"]],
                        "bbox": bbox,
                        "content": text_content,
                        "source_objects": [b.id for b in blocks_sorted],
                        "references": [],
                    }
                )

    return _merge_cross_page_figures(chunks, page_heights)
