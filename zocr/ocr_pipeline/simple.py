"""Minimal built-in implementations for the OCR pipeline.

These classes are intentionally lightweight so the pipeline can run without
external ML models while still exercising the end-to-end flow on real images.
"""
from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request
from bisect import bisect_right
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, TypedDict

import numpy as np
from PIL import Image

try:  # pragma: no cover - optional dependency
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover - optional dependency missing
    pytesseract = None  # type: ignore

from .interfaces import Aggregator, RegionClassifier, Segmenter, TableExtractor, VLLM
from .models import (
    BoundingBox,
    ClassifiedRegion,
    DocumentMetadata,
    DocumentOutput,
    ImageCaptionResult,
    PageInput,
    RegionOutput,
    RegionType,
    SegmentedRegion,
    TableData,
    TableExtractionResult,
    TextOcrResult,
)
from .structure import build_structural_graph

LineKey = Tuple[int, int, int]
XYBox = Tuple[int, int, int, int]


class OcrWord(TypedDict):
    text: str
    left: int
    top: int
    width: int
    height: int
    conf: float
    row_key: LineKey


def _to_gray_array(image: Image.Image) -> np.ndarray:
    gray = image.convert("L")
    return np.array(gray)


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    if gray.size == 0:
        return np.zeros_like(gray, dtype=bool)
    percentile = float(np.percentile(gray, 70))
    otsu = _otsu_threshold(gray)
    threshold = percentile - 12.0
    if 10.0 < otsu < 245.0:
        threshold = min(threshold, otsu)
    threshold = max(0.0, min(255.0, threshold))
    return gray <= threshold


def _otsu_threshold(gray: np.ndarray) -> float:
    if gray.size == 0:
        return 0.0
    hist = np.bincount(gray.reshape(-1), minlength=256).astype(float)
    total = hist.sum()
    if total <= 0:
        return 0.0
    cumulative = np.cumsum(hist)
    cumulative_mean = np.cumsum(hist * np.arange(256))
    global_mean = cumulative_mean[-1]
    denominator = cumulative * (total - cumulative)
    valid = denominator > 0
    variance = np.zeros(256, dtype=float)
    variance[valid] = (
        (global_mean * cumulative[valid] - cumulative_mean[valid] * total) ** 2
        / denominator[valid]
    )
    return float(np.argmax(variance))


def _find_gaps(
    density: np.ndarray, gap_threshold: float, min_gap_px: int
) -> List[Tuple[int, int]]:
    gaps: List[Tuple[int, int]] = []
    start = None
    for idx, value in enumerate(density):
        if value <= gap_threshold:
            if start is None:
                start = idx
        else:
            if start is not None and idx - start >= min_gap_px:
                gaps.append((start, idx))
            start = None
    if start is not None and len(density) - start >= min_gap_px:
        gaps.append((start, len(density)))
    return gaps


def _smooth_density(density: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return density
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(density, kernel, mode="same")


def _segments_from_gaps(length: int, gaps: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not gaps:
        return [(0, length)]
    segments: List[Tuple[int, int]] = []
    cursor = 0
    for start, end in gaps:
        if cursor < start:
            segments.append((cursor, start))
        cursor = end
    if cursor < length:
        segments.append((cursor, length))
    return segments


def _bounding_box_from_mask(
    mask: np.ndarray, x_offset: int, y_offset: int, min_size: int
) -> Tuple[int, int, int, int]:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return x_offset, y_offset, min_size, min_size
    y_indices = np.where(rows)[0]
    x_indices = np.where(cols)[0]
    y0, y1 = int(y_indices[0]), int(y_indices[-1]) + 1
    x0, x1 = int(x_indices[0]), int(x_indices[-1]) + 1
    width = max(min_size, x1 - x0)
    height = max(min_size, y1 - y0)
    return x_offset + x0, y_offset + y0, width, height


def _ink_bounds(mask: np.ndarray) -> Optional[XYBox]:
    if mask.size == 0 or not mask.any():
        return None
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None
    y_indices = np.where(rows)[0]
    x_indices = np.where(cols)[0]
    return (
        int(x_indices[0]),
        int(y_indices[0]),
        int(x_indices[-1]) + 1,
        int(y_indices[-1]) + 1,
    )


def _pad_xy_box(box: XYBox, width: int, height: int, padding: int) -> XYBox:
    x0, y0, x1, y1 = box
    return (
        max(0, x0 - padding),
        max(0, y0 - padding),
        min(width, x1 + padding),
        min(height, y1 + padding),
    )


def _tighten_xy_box_to_ink(mask: np.ndarray, box: XYBox, padding: int = 0) -> Optional[XYBox]:
    x0, y0, x1, y1 = box
    x0 = max(0, min(int(x0), mask.shape[1]))
    x1 = max(0, min(int(x1), mask.shape[1]))
    y0 = max(0, min(int(y0), mask.shape[0]))
    y1 = max(0, min(int(y1), mask.shape[0]))
    if x1 <= x0 or y1 <= y0:
        return None
    local = _ink_bounds(mask[y0:y1, x0:x1])
    if local is None:
        return None
    lx0, ly0, lx1, ly1 = local
    return _pad_xy_box(
        (x0 + lx0, y0 + ly0, x0 + lx1, y0 + ly1),
        mask.shape[1],
        mask.shape[0],
        padding,
    )


def _valid_split_gap(start: int, end: int, axis_len: int, min_side_px: int) -> bool:
    if start <= 0 or end >= axis_len:
        return False
    return start >= min_side_px and axis_len - end >= min_side_px


def _best_gap(
    density: np.ndarray,
    *,
    gap_threshold: float,
    min_gap_px: int,
    min_side_px: int,
) -> Optional[Tuple[int, int, float]]:
    candidates = []
    for start, end in _find_gaps(density, gap_threshold, min_gap_px):
        if not _valid_split_gap(start, end, len(density), min_side_px):
            continue
        gap_len = max(1, end - start)
        balance = min(start, len(density) - end) / max(1, len(density))
        candidates.append((start, end, (gap_len / max(1, len(density))) + balance * 0.25))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[2])


def _split_box_xy(
    mask: np.ndarray,
    box: XYBox,
    *,
    gap_ratio: float,
    min_gap_fraction: float,
    min_region_size: int,
    smoothing_window: int,
    max_depth: int,
    bbox_padding: int = 0,
    depth: int = 0,
) -> List[XYBox]:
    tightened = _tighten_xy_box_to_ink(mask, box, padding=bbox_padding)
    if tightened is None:
        return []
    x0, y0, x1, y1 = tightened
    box_w = x1 - x0
    box_h = y1 - y0
    if depth >= max_depth or box_w < min_region_size * 2 or box_h < min_region_size * 2:
        return [tightened]

    crop = mask[y0:y1, x0:x1]
    row_density = _smooth_density(crop.mean(axis=1), min(smoothing_window, max(3, box_h // 40)))
    col_density = _smooth_density(crop.mean(axis=0), min(smoothing_window, max(3, box_w // 40)))
    row_gap = _best_gap(
        row_density,
        gap_threshold=gap_ratio,
        min_gap_px=max(4, int(box_h * min_gap_fraction)),
        min_side_px=min_region_size,
    )
    col_gap = _best_gap(
        col_density,
        gap_threshold=gap_ratio,
        min_gap_px=max(4, int(box_w * min_gap_fraction)),
        min_side_px=min_region_size,
    )

    split_axis: Optional[str] = None
    split_gap: Optional[Tuple[int, int, float]] = None
    if row_gap and col_gap:
        row_score = row_gap[2] * (1.15 if box_w >= box_h else 1.0)
        col_score = col_gap[2] * (1.15 if box_h >= box_w else 1.0)
        split_axis, split_gap = ("rows", row_gap) if row_score >= col_score else ("cols", col_gap)
    elif row_gap:
        split_axis, split_gap = "rows", row_gap
    elif col_gap:
        split_axis, split_gap = "cols", col_gap

    if split_axis is None or split_gap is None:
        return [tightened]

    start, end, _score = split_gap
    if split_axis == "rows":
        children = [(x0, y0, x1, y0 + start), (x0, y0 + end, x1, y1)]
    else:
        children = [(x0, y0, x0 + start, y1), (x0 + end, y0, x1, y1)]

    leaves: List[XYBox] = []
    for child in children:
        leaves.extend(
            _split_box_xy(
                mask,
                child,
                gap_ratio=gap_ratio,
                min_gap_fraction=min_gap_fraction,
                min_region_size=min_region_size,
                smoothing_window=smoothing_window,
                max_depth=max_depth,
                bbox_padding=bbox_padding,
                depth=depth + 1,
            )
        )
    return leaves or [tightened]


def _xy_box_iou(a: XYBox, b: XYBox) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    intersection = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(1, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1, (bx1 - bx0) * (by1 - by0))
    return float(intersection / max(1, area_a + area_b - intersection))


def _dedupe_xy_boxes(boxes: Sequence[XYBox], iou_threshold: float = 0.92) -> List[XYBox]:
    deduped: List[XYBox] = []
    for box in boxes:
        if any(_xy_box_iou(box, existing) >= iou_threshold for existing in deduped):
            continue
        deduped.append(box)
    return deduped


def _clip_box(
    x: int, y: int, width: int, height: int, image_width: int, image_height: int
) -> Tuple[int, int, int, int]:
    if image_width <= 0 or image_height <= 0:
        return 0, 0, 0, 0
    x0 = max(0, min(image_width - 1, x))
    y0 = max(0, min(image_height - 1, y))
    x1 = max(x0 + 1, min(image_width, x + width))
    y1 = max(y0 + 1, min(image_height, y + height))
    return x0, y0, x1 - x0, y1 - y0


def _pad_box(
    box: Tuple[int, int, int, int], padding: int, image_width: int, image_height: int
) -> Tuple[int, int, int, int]:
    x, y, width, height = box
    return _clip_box(
        x - padding,
        y - padding,
        width + padding * 2,
        height + padding * 2,
        image_width,
        image_height,
    )


def _coarse_mask(mask: np.ndarray, cell_size: int) -> np.ndarray:
    if mask.size == 0:
        return np.zeros((0, 0), dtype=bool)
    height, width = mask.shape
    rows = int(np.ceil(height / cell_size))
    cols = int(np.ceil(width / cell_size))
    padded = np.zeros((rows * cell_size, cols * cell_size), dtype=bool)
    padded[:height, :width] = mask
    return padded.reshape(rows, cell_size, cols, cell_size).any(axis=(1, 3))


def _dilate_bool(mask: np.ndarray, radius_y: int, radius_x: int) -> np.ndarray:
    if mask.size == 0:
        return mask
    radius_y = max(0, radius_y)
    radius_x = max(0, radius_x)
    padded = np.pad(mask, ((radius_y, radius_y), (radius_x, radius_x)), constant_values=False)
    out = np.zeros_like(mask, dtype=bool)
    for dy in range(radius_y * 2 + 1):
        for dx in range(radius_x * 2 + 1):
            out |= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return out


def _component_boxes_from_mask(
    mask: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    cell_size: int,
    dilation: int,
    min_component_cells: int,
    padding: int,
) -> List[Tuple[int, int, int, int]]:
    coarse = _dilate_bool(_coarse_mask(mask, cell_size), dilation, dilation)
    if coarse.size == 0:
        return []

    visited = np.zeros_like(coarse, dtype=bool)
    boxes: List[Tuple[int, int, int, int]] = []
    rows, cols = coarse.shape
    for row in range(rows):
        for col in range(cols):
            if visited[row, col] or not coarse[row, col]:
                continue
            stack = [(row, col)]
            visited[row, col] = True
            min_row = max_row = row
            min_col = max_col = col
            count = 0
            while stack:
                current_row, current_col = stack.pop()
                count += 1
                min_row = min(min_row, current_row)
                max_row = max(max_row, current_row)
                min_col = min(min_col, current_col)
                max_col = max(max_col, current_col)
                for next_row, next_col in (
                    (current_row - 1, current_col),
                    (current_row + 1, current_col),
                    (current_row, current_col - 1),
                    (current_row, current_col + 1),
                ):
                    if next_row < 0 or next_col < 0 or next_row >= rows or next_col >= cols:
                        continue
                    if visited[next_row, next_col] or not coarse[next_row, next_col]:
                        continue
                    visited[next_row, next_col] = True
                    stack.append((next_row, next_col))
            if count < min_component_cells:
                continue
            x = min_col * cell_size
            y = min_row * cell_size
            width = (max_col - min_col + 1) * cell_size
            height = (max_row - min_row + 1) * cell_size
            boxes.append(_pad_box((x, y, width, height), padding, image_width, image_height))
    return sorted(boxes, key=lambda item: (item[1], item[0], item[3], item[2]))


class FullPageSegmenter(Segmenter):
    """Split the page into regions using whitespace plus connected components."""

    def __init__(
        self,
        confidence: float = 0.85,
        gap_ratio: float = 0.015,
        min_gap_fraction: float = 0.02,
        min_region_fraction: float = 0.08,
        smoothing_window: int = 7,
        bbox_padding: int = 3,
        component_cell_size: int = 12,
        component_dilation: int = 1,
        min_component_cells: int = 2,
        max_depth: int = 6,
    ) -> None:
        self.confidence = confidence
        self.gap_ratio = gap_ratio
        self.min_gap_fraction = min_gap_fraction
        self.min_region_fraction = min_region_fraction
        self.smoothing_window = smoothing_window
        self.bbox_padding = bbox_padding
        self.component_cell_size = component_cell_size
        self.component_dilation = component_dilation
        self.min_component_cells = min_component_cells
        self.max_depth = max_depth

    def segment(self, page: PageInput) -> List[SegmentedRegion]:
        image = page.image
        if not isinstance(image, Image.Image):
            raise TypeError("FullPageSegmenter expects a PIL.Image instance")

        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError("Page image must have positive dimensions")

        gray = _to_gray_array(image)
        mask = _ink_mask(gray)
        regions: List[SegmentedRegion] = []
        min_region_size = max(8, int(min(width, height) * self.min_region_fraction))
        if mask.any():
            content_box = _tighten_xy_box_to_ink(
                mask,
                (0, 0, width, height),
                padding=self.bbox_padding,
            )
            boxes = (
                _split_box_xy(
                    mask,
                    content_box,
                    gap_ratio=self.gap_ratio,
                    min_gap_fraction=self.min_gap_fraction,
                    min_region_size=min_region_size,
                    smoothing_window=self.smoothing_window,
                    max_depth=self.max_depth,
                    bbox_padding=self.bbox_padding,
                )
                if content_box
                else []
            )
            for region_index, (x0, y0, x1, y1) in enumerate(
                _dedupe_xy_boxes(boxes),
                start=1,
            ):
                region_w = max(1, x1 - x0)
                region_h = max(1, y1 - y0)
                if region_w < min_region_size and region_h < min_region_size:
                    continue
                region_id = f"{page.document_id}-page{page.page_number}-region{region_index}"
                bbox = BoundingBox(x=x0, y=y0, width=region_w, height=region_h)
                crop = image.crop((x0, y0, x1, y1))
                regions.append(
                    SegmentedRegion(
                        region_id=region_id,
                        bounding_box=bbox,
                        image_crop=crop,
                        confidence=self.confidence,
                        reading_order=len(regions),
                    )
                )

        if len(regions) <= 1:
            component_regions = self._component_regions(
                page=page,
                image=image,
                mask=mask,
                image_width=width,
                image_height=height,
                min_region_size=min_region_size,
            )
            if len(component_regions) > len(regions):
                regions = component_regions

        if not regions:
            bounding_box = BoundingBox(x=0, y=0, width=width, height=height)
            region_id = f"{page.document_id}-page{page.page_number}-full"
            regions = [
                SegmentedRegion(
                    region_id=region_id,
                    bounding_box=bounding_box,
                    image_crop=image,
                    confidence=self.confidence,
                    reading_order=0,
                )
            ]

        return regions

    def _component_regions(
        self,
        *,
        page: PageInput,
        image: Image.Image,
        mask: np.ndarray,
        image_width: int,
        image_height: int,
        min_region_size: int,
    ) -> List[SegmentedRegion]:
        cell_size = max(4, min(self.component_cell_size, max(4, min(image_width, image_height) // 8)))
        boxes = _component_boxes_from_mask(
            mask,
            image_width=image_width,
            image_height=image_height,
            cell_size=cell_size,
            dilation=self.component_dilation,
            min_component_cells=self.min_component_cells,
            padding=max(self.bbox_padding, cell_size // 2),
        )
        regions: List[SegmentedRegion] = []
        for box in boxes:
            x, y, region_w, region_h = box
            if region_w < min_region_size and region_h < min_region_size:
                continue
            region_id = f"{page.document_id}-page{page.page_number}-component{len(regions) + 1}"
            bbox = BoundingBox(x=x, y=y, width=region_w, height=region_h)
            crop = image.crop((x, y, x + region_w, y + region_h))
            regions.append(
                SegmentedRegion(
                    region_id=region_id,
                    bounding_box=bbox,
                    image_crop=crop,
                    confidence=max(0.0, self.confidence - 0.05),
                    reading_order=len(regions),
                )
            )
        return regions


def _edge_density(gray: np.ndarray) -> float:
    if gray.size == 0:
        return 0.0
    diff_x = np.abs(np.diff(gray.astype(np.int16), axis=1))
    diff_y = np.abs(np.diff(gray.astype(np.int16), axis=0))
    edges = (diff_x > 20).mean() + (diff_y > 20).mean()
    return float(edges / 2.0)


def _line_ratios(mask: np.ndarray) -> Tuple[float, float]:
    if mask.size == 0:
        return 0.0, 0.0
    row_density = mask.mean(axis=1)
    col_density = mask.mean(axis=0)
    row_ratio = float((row_density > 0.6).mean())
    col_ratio = float((col_density > 0.6).mean())
    return row_ratio, col_ratio


class AspectRatioRegionClassifier(RegionClassifier):
    """Classify regions using geometry and lightweight image statistics."""

    def __init__(
        self,
        table_aspect_ratio: float = 2.5,
        square_tolerance: float = 0.15,
        default_type: RegionType = RegionType.TEXT,
    ) -> None:
        self.table_aspect_ratio = table_aspect_ratio
        self.square_tolerance = square_tolerance
        self.default_type = default_type

    def classify(self, region: SegmentedRegion) -> ClassifiedRegion:
        width = region.bounding_box.width
        height = region.bounding_box.height
        ratio = width / height if height else 0

        classification = self.default_type
        confidence = region.confidence
        if isinstance(region.image_crop, Image.Image):
            gray = _to_gray_array(region.image_crop)
            mask = _ink_mask(gray)
            ink_ratio = float(mask.mean()) if mask.size else 0.0
            edge_ratio = _edge_density(gray)
            row_ratio, col_ratio = _line_ratios(mask)

            if row_ratio > 0.03 and col_ratio > 0.02 and 0.01 < ink_ratio < 0.45:
                classification = RegionType.TABLE
                confidence = min(1.0, confidence + 0.1)
            elif ink_ratio < 0.02 and edge_ratio > 0.06:
                classification = RegionType.IMAGE
                confidence = min(1.0, confidence + 0.05)
            elif ratio >= self.table_aspect_ratio:
                classification = RegionType.TABLE
            elif abs(1 - ratio) <= self.square_tolerance:
                classification = RegionType.IMAGE
        else:
            if ratio >= self.table_aspect_ratio:
                classification = RegionType.TABLE
            elif abs(1 - ratio) <= self.square_tolerance:
                classification = RegionType.IMAGE

        return ClassifiedRegion(
            region_id=region.region_id,
            bounding_box=region.bounding_box,
            classification=classification,
            confidence=confidence,
            reading_order=region.reading_order,
            image_crop=region.image_crop,
        )


class SimpleAggregator(Aggregator):
    """Aggregate component outputs into a document payload."""

    def aggregate(
        self,
        page: PageInput,
        classified_regions: List[ClassifiedRegion],
        text_results: List[TextOcrResult],
        image_results: List[ImageCaptionResult],
        table_results: List[TableExtractionResult],
    ) -> DocumentOutput:
        text_map = {item.region_id: item for item in text_results}
        image_map = {item.region_id: item for item in image_results}
        table_map = {item.region_id: item for item in table_results}

        sorted_regions = sorted(
            classified_regions,
            key=lambda r: (
                r.reading_order if r.reading_order is not None else float("inf"),
                r.region_id,
            ),
        )

        region_outputs: List[RegionOutput] = []
        for region in sorted_regions:
            if region.classification == RegionType.TEXT and region.region_id in text_map:
                content = {
                    "text": text_map[region.region_id].text,
                    "confidence": text_map[region.region_id].confidence,
                    "language": text_map[region.region_id].language,
                }
            elif region.classification == RegionType.IMAGE and region.region_id in image_map:
                content = {
                    "caption": image_map[region.region_id].caption,
                    "confidence": image_map[region.region_id].confidence,
                    "detailed_description": image_map[region.region_id].detailed_description,
                    "detected_objects": image_map[region.region_id].detected_objects,
                }
            elif region.classification == RegionType.TABLE and region.region_id in table_map:
                content = {
                    "table_data": table_map[region.region_id].table_data.model_dump(),
                    "confidence": table_map[region.region_id].confidence,
                    "format": table_map[region.region_id].format,
                }
            else:
                content = {}

            region_outputs.append(
                RegionOutput(
                    region_id=region.region_id,
                    type=region.classification,
                    bounding_box=region.bounding_box,
                    reading_order=region.reading_order,
                    content=content,
                )
            )

        metadata = DocumentMetadata(
            total_regions=len(region_outputs),
            text_regions=sum(1 for region in region_outputs if region.type == RegionType.TEXT),
            image_regions=sum(1 for region in region_outputs if region.type == RegionType.IMAGE),
            table_regions=sum(1 for region in region_outputs if region.type == RegionType.TABLE),
        )

        structure = build_structural_graph(
            document_id=page.document_id,
            page_number=page.page_number,
            regions=region_outputs,
        )

        return DocumentOutput(
            document_id=page.document_id,
            page_number=page.page_number,
            regions=region_outputs,
            metadata=metadata,
            structure=structure,
        )


def _dominant_color(image: Image.Image) -> str:
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")
    small = image.resize((32, 32))
    data = np.array(small)[:, :, :3].reshape(-1, 3)
    mean = data.mean(axis=0)
    return f"rgb({int(mean[0])},{int(mean[1])},{int(mean[2])})"


def _colorfulness(image: Image.Image) -> float:
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")
    small = image.resize((32, 32))
    data = np.array(small)[:, :, :3].astype(float)
    rg = np.abs(data[:, :, 0] - data[:, :, 1])
    yb = np.abs(0.5 * (data[:, :, 0] + data[:, :, 1]) - data[:, :, 2])
    return float(
        np.sqrt(np.var(rg) + np.var(yb))
        + 0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
    )


CaptionerResult = ImageCaptionResult | Mapping[str, Any] | str | None
Captioner = Callable[[ClassifiedRegion], CaptionerResult]


def _image_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _caption_result_from_provider(
    region: ClassifiedRegion,
    payload: CaptionerResult,
    *,
    default_confidence: float,
) -> Optional[ImageCaptionResult]:
    if payload is None:
        return None
    if isinstance(payload, ImageCaptionResult):
        return payload
    if isinstance(payload, str):
        caption = payload.strip()
        if not caption:
            return None
        return ImageCaptionResult(
            region_id=region.region_id,
            caption=caption,
            confidence=default_confidence,
        )
    caption = str(payload.get("caption") or payload.get("text") or "").strip()
    if not caption:
        return None
    confidence_raw = payload.get("confidence", default_confidence)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = default_confidence
    detected = payload.get("detected_objects")
    if detected is not None and not isinstance(detected, list):
        detected = [str(detected)]
    description = payload.get("detailed_description") or payload.get("description")
    return ImageCaptionResult(
        region_id=str(payload.get("region_id") or region.region_id),
        caption=caption,
        confidence=max(0.0, min(1.0, confidence)),
        detected_objects=detected,
        detailed_description=str(description) if description else None,
    )


class SimpleVisualDescriptor(VLLM):
    """Local visual descriptor for image-like regions.

    This is deliberately offline and deterministic. It does not claim semantic
    object recognition, but it produces richer cues than a bare placeholder so
    downstream review can distinguish photos, screenshots, diagrams, and flat
    graphics without adding a required model dependency.
    """

    def __init__(self, confidence: float = 0.55, captioner: Captioner | None = None) -> None:
        self.confidence = confidence
        self.captioner = captioner

    def describe(self, region: ClassifiedRegion) -> ImageCaptionResult:
        caption_parts: List[str] = []
        detected: List[str] = []
        detail: List[str] = []

        if self.captioner is not None:
            try:
                provided = _caption_result_from_provider(
                    region,
                    self.captioner(region),
                    default_confidence=self.confidence,
                )
            except Exception as exc:
                provided = None
                detail.append(f"caption provider failed: {exc}")
            if provided is not None:
                return provided

        if isinstance(region.image_crop, Image.Image):
            image = region.image_crop
            width, height = image.size
            gray = _to_gray_array(image)
            mask = _ink_mask(gray)
            ink_ratio = float(mask.mean()) if mask.size else 0.0
            edge_ratio = _edge_density(gray)
            row_ratio, col_ratio = _line_ratios(mask)
            variance = float(np.var(gray)) if gray.size else 0.0
            dominant = _dominant_color(image)
            colorfulness = _colorfulness(image)
            aspect = width / height if height else 0.0

            caption_parts.append(f"Visual region {width}x{height}")
            detail.append(f"dominant color {dominant}")
            detail.append(f"aspect ratio {aspect:.2f}")
            detail.append(f"colorfulness {colorfulness:.1f}")
            detail.append(f"edge density {edge_ratio:.2f}")

            if row_ratio > 0.03 and col_ratio > 0.02:
                caption_parts.append("grid or table-like graphic")
                detected.append("table_grid")
            elif ink_ratio < 0.02 and edge_ratio > 0.07 and colorfulness > 8.0:
                caption_parts.append("photo-like raster content")
                detected.append("photo_like")
            elif ink_ratio > 0.05 and edge_ratio > 0.05:
                caption_parts.append("line-art diagram or chart")
                detected.append("diagram_like")
            elif edge_ratio > 0.08 and colorfulness < 6.0:
                caption_parts.append("screenshot or document crop")
                detected.append("screenshot_like")
            elif variance < 120:
                caption_parts.append("flat graphic")
                detected.append("flat_graphic")
            else:
                caption_parts.append("mixed visual content")
            detail.append(f"ink ratio {ink_ratio:.3f}")
            detail.append(f"line ratios h={row_ratio:.3f} v={col_ratio:.3f}")
            detail.append(f"variance {variance:.1f}")
        else:
            caption_parts.append(f"Visual region {region.region_id}")

        return ImageCaptionResult(
            region_id=region.region_id,
            caption="; ".join(caption_parts),
            confidence=self.confidence,
            detected_objects=detected or None,
            detailed_description=", ".join(detail) if detail else None,
        )


class SimpleVLLM(SimpleVisualDescriptor):
    """Backward-compatible alias for the legacy VLM-like name."""


class HttpVLLM(SimpleVisualDescriptor):
    """HTTP-backed VLM adapter with the same result contract as ``SimpleVisualDescriptor``."""

    def __init__(
        self,
        endpoint: str,
        *,
        timeout_s: float = 20.0,
        confidence: float = 0.55,
    ) -> None:
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        super().__init__(confidence=confidence, captioner=self._describe_with_endpoint)

    def _describe_with_endpoint(self, region: ClassifiedRegion) -> CaptionerResult:
        if not isinstance(region.image_crop, Image.Image):
            return None
        image = region.image_crop
        payload = {
            "region_id": region.region_id,
            "image": _image_data_url(image),
            "mime_type": "image/png",
            "metadata": {
                "width": image.size[0],
                "height": image.size[1],
                "bounding_box": region.bounding_box.model_dump(),
                "classification": region.classification.value,
            },
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"VLM endpoint unavailable: {exc}") from exc
        return json.loads(raw)


class DummyVLLM(SimpleVisualDescriptor):
    """Backward-compatible alias for the heuristic visual descriptor."""


def _cluster_centers(values: Iterable[float], gap: float) -> List[float]:
    sorted_values = sorted(values)
    if not sorted_values:
        return []
    centers = [sorted_values[0]]
    counts = [1]
    for value in sorted_values[1:]:
        if abs(value - centers[-1]) <= gap:
            counts[-1] += 1
            centers[-1] = centers[-1] + (value - centers[-1]) / counts[-1]
        else:
            centers.append(value)
            counts.append(1)
    return centers


def _is_header_candidate(cells: Sequence[str]) -> bool:
    if not cells:
        return False
    alpha_cells = sum(1 for cell in cells if any(ch.isalpha() for ch in cell))
    return alpha_cells >= max(1, len(cells) // 2)


def _normalize_tesseract_conf(raw: Any) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return -1.0
    if value < 0:
        return -1.0
    return max(0.0, min(1.0, value / 100.0 if value > 1.0 else value))


def _data_value(data: Mapping[str, Sequence[Any]], key: str, idx: int, default: Any = 0) -> Any:
    values = data.get(key, [])
    try:
        return values[idx]
    except Exception:
        return default


def _word_rows(words: Sequence[OcrWord]) -> List[List[OcrWord]]:
    if not words:
        return []
    heights = [float(w["height"]) for w in words if float(w.get("height", 0)) > 0]
    median_height = float(np.median(heights)) if heights else 10.0
    row_gap = max(6.0, median_height * 0.85)
    centers = _cluster_centers(
        (float(w["top"]) + float(w["height"]) / 2.0 for w in words),
        row_gap,
    )
    centers = sorted(centers) or [0.0]
    rows: Dict[int, List[OcrWord]] = {}
    for word in words:
        center = float(word["top"]) + float(word["height"]) / 2.0
        row_idx = min(range(len(centers)), key=lambda i: abs(centers[i] - center))
        rows.setdefault(row_idx, []).append(word)
    return [sorted(rows[key], key=lambda w: (int(w["left"]), int(w["top"]))) for key in sorted(rows)]


def _row_cell_starts(rows: Sequence[Sequence[OcrWord]]) -> List[float]:
    widths = [float(w["width"]) for row in rows for w in row if float(w.get("width", 0)) > 0]
    median_width = float(np.median(widths)) if widths else 20.0
    gap_threshold = max(10.0, median_width * 0.75)
    starts: List[float] = []
    for row in rows:
        ordered = sorted(row, key=lambda w: int(w["left"]))
        if not ordered:
            continue
        starts.append(float(ordered[0]["left"]))
        last_right = float(ordered[0]["left"] + ordered[0]["width"])
        for word in ordered[1:]:
            left = float(word["left"])
            gap = left - last_right
            if gap >= gap_threshold:
                starts.append(left)
            last_right = max(last_right, left + float(word["width"]))
    return starts


def _infer_column_anchors(rows: Sequence[Sequence[OcrWord]]) -> List[float]:
    starts = _row_cell_starts(rows)
    widths = [float(w["width"]) for row in rows for w in row if float(w.get("width", 0)) > 0]
    median_width = float(np.median(widths)) if widths else 20.0
    start_gap = max(10.0, median_width * 0.9)
    anchors = _cluster_centers(starts, start_gap)
    if len(anchors) <= 1:
        center_gap = max(12.0, median_width * 1.6)
        anchors = _cluster_centers(
            (float(w["left"]) + float(w["width"]) / 2.0 for row in rows for w in row),
            center_gap,
        )
    return sorted(anchors) or [0.0]


def _assign_word_to_column(word: Mapping[str, Any], anchors: Sequence[float]) -> int:
    if len(anchors) <= 1:
        return 0
    boundaries = [(anchors[idx] + anchors[idx + 1]) / 2.0 for idx in range(len(anchors) - 1)]
    center = float(word["left"]) + float(word["width"]) / 2.0
    return min(len(anchors) - 1, bisect_right(boundaries, center))


def _unique_headers(cells: Sequence[str]) -> List[str]:
    headers: List[str] = []
    seen: Dict[str, int] = {}
    for idx, cell in enumerate(cells):
        base = cell.strip() or f"col{idx + 1}"
        count = seen.get(base, 0)
        seen[base] = count + 1
        headers.append(base if count == 0 else f"{base}_{count + 1}")
    return headers


def _table_data_from_rows(rows: List[List[str]], column_count: int) -> TableData:
    if column_count <= 0:
        column_count = max((len(row) for row in rows), default=1)
    normalized_rows = [
        [row[idx] if idx < len(row) else "" for idx in range(column_count)] for row in rows
    ]
    if normalized_rows and _is_header_candidate(normalized_rows[0]):
        header_cells = _unique_headers(normalized_rows[0])
        data_rows = normalized_rows[1:]
    else:
        header_cells = [f"col{idx + 1}" for idx in range(column_count)]
        data_rows = normalized_rows

    row_dicts = [
        {header: row[idx] if idx < len(row) else "" for idx, header in enumerate(header_cells)}
        for row in data_rows
    ]
    return TableData(
        headers=header_cells,
        rows=row_dicts,
        num_rows=len(row_dicts),
        num_columns=len(header_cells),
    )


def _dense_runs(density: np.ndarray, threshold: float, min_width: int) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    start = None
    for idx, value in enumerate(density):
        if value >= threshold:
            if start is None:
                start = idx
        else:
            if start is not None and idx - start >= min_width:
                runs.append((start, idx))
            start = None
    if start is not None and len(density) - start >= min_width:
        runs.append((start, len(density)))
    return runs


def _run_centers(runs: Sequence[Tuple[int, int]]) -> List[int]:
    return [int(round((start + end - 1) / 2.0)) for start, end in runs]


def _grid_boundaries(mask: np.ndarray) -> Tuple[List[int], List[int]]:
    if mask.size == 0:
        return [], []
    height, width = mask.shape
    horizontal_runs = _dense_runs(
        mask.mean(axis=1),
        threshold=0.55,
        min_width=max(1, int(height * 0.002)),
    )
    vertical_runs = _dense_runs(
        mask.mean(axis=0),
        threshold=0.55,
        min_width=max(1, int(width * 0.002)),
    )
    horizontal = _run_centers(horizontal_runs)
    vertical = _run_centers(vertical_runs)
    if len(horizontal) < 2 or len(vertical) < 2:
        return [], []
    return horizontal, vertical


def _assign_words_to_grid(
    words: Sequence[OcrWord], horizontal: Sequence[int], vertical: Sequence[int]
) -> List[List[str]]:
    row_count = len(horizontal) - 1
    col_count = len(vertical) - 1
    rows: List[List[List[str]]] = [[[] for _ in range(col_count)] for _ in range(row_count)]
    for word in sorted(words, key=lambda item: (item["top"], item["left"])):
        center_x = word["left"] + word["width"] / 2.0
        center_y = word["top"] + word["height"] / 2.0
        row_idx = next(
            (
                idx
                for idx in range(row_count)
                if horizontal[idx] <= center_y <= horizontal[idx + 1]
            ),
            None,
        )
        col_idx = next(
            (
                idx
                for idx in range(col_count)
                if vertical[idx] <= center_x <= vertical[idx + 1]
            ),
            None,
        )
        if row_idx is None or col_idx is None:
            continue
        rows[row_idx][col_idx].append(word["text"])
    return [[" ".join(cell).strip() for cell in row] for row in rows]


class SimpleTableExtractor(TableExtractor):
    """Extract tables using grid lines when present, then OCR word clustering."""

    def __init__(self, confidence: float = 0.6) -> None:
        self.confidence = confidence

    def extract(self, region: ClassifiedRegion) -> TableExtractionResult:
        if not isinstance(region.image_crop, Image.Image):
            table_data = TableData(headers=["col1"], rows=[], num_rows=0, num_columns=1)
            return TableExtractionResult(
                region_id=region.region_id,
                table_data=table_data,
                confidence=0.2,
                format="empty",
            )

        gray = _to_gray_array(region.image_crop)
        mask = _ink_mask(gray)
        horizontal, vertical = _grid_boundaries(mask)
        words: List[OcrWord] = []

        if pytesseract is None:
            if horizontal and vertical:
                row_count = len(horizontal) - 1
                col_count = len(vertical) - 1
                table_data = _table_data_from_rows(
                    [["" for _ in range(col_count)] for _ in range(row_count)],
                    col_count,
                )
                return TableExtractionResult(
                    region_id=region.region_id,
                    table_data=table_data,
                    confidence=0.35,
                    format="grid_no_ocr",
                )
            table_data = TableData(headers=["col1"], rows=[], num_rows=0, num_columns=1)
            return TableExtractionResult(
                region_id=region.region_id,
                table_data=table_data,
                confidence=0.0,
                format="missing_pytesseract",
            )

        try:
            data = pytesseract.image_to_data(region.image_crop, output_type=pytesseract.Output.DICT)
        except Exception:
            table_data = TableData(headers=["col1"], rows=[], num_rows=0, num_columns=1)
            return TableExtractionResult(
                region_id=region.region_id,
                table_data=table_data,
                confidence=0.0,
                format="tesseract_error",
            )
        texts = data.get("text", [])
        for idx, text in enumerate(texts):
            if not text or text.strip() == "":
                continue
            conf_val = _normalize_tesseract_conf(_data_value(data, "conf", idx, -1.0))
            if conf_val < 0:
                continue
            left = int(_data_value(data, "left", idx, 0))
            top = int(_data_value(data, "top", idx, 0))
            width = int(_data_value(data, "width", idx, 0))
            height = int(_data_value(data, "height", idx, 0))
            block_num = int(_data_value(data, "block_num", idx, 0))
            par_num = int(_data_value(data, "par_num", idx, 0))
            line_num = int(_data_value(data, "line_num", idx, 0))
            words.append(
                {
                    "text": text.strip(),
                    "left": left,
                    "top": top,
                    "width": width,
                    "height": height,
                    "conf": conf_val,
                    "row_key": (block_num, par_num, line_num),
                }
            )

        if horizontal and vertical:
            grid_rows = _assign_words_to_grid(words, horizontal, vertical)
            table_data = _table_data_from_rows(grid_rows, len(vertical) - 1)
            mean_word_conf = float(np.mean([word["conf"] for word in words])) if words else 0.0
            confidence = min(
                0.95,
                max(
                    self.confidence,
                    mean_word_conf * 0.85 + 0.12 + (0.03 * min(table_data.num_rows or 0, 4)),
                ),
            )
            return TableExtractionResult(
                region_id=region.region_id,
                table_data=table_data,
                confidence=confidence,
                format="tesseract_grid",
            )

        if not words:
            table_data = TableData(headers=["col1"], rows=[], num_rows=0, num_columns=1)
            return TableExtractionResult(
                region_id=region.region_id,
                table_data=table_data,
                confidence=0.25,
                format="empty",
            )

        words.sort(key=lambda w: (w["top"], w["left"]))
        ordered_lines = _word_rows(words)
        centers = _infer_column_anchors(ordered_lines)
        rows: List[List[str]] = []
        row_confidences: List[List[float]] = []
        for line_words in ordered_lines:
            cells = ["" for _ in centers]
            conf_cells: List[List[float]] = [[] for _ in centers]
            for word in line_words:
                column_idx = _assign_word_to_column(word, centers)
                cells[column_idx] = (cells[column_idx] + " " + word["text"]).strip()
                conf_cells[column_idx].append(float(word.get("conf", 0.0)))
            rows.append(cells)
            row_confidences.append(
                [float(np.mean(values)) if values else 0.0 for values in conf_cells]
            )

        table_data = _table_data_from_rows(rows, len(centers))
        observed_conf = [
            conf
            for row in row_confidences
            for conf in row
            if conf > 0
        ] or [float(word.get("conf", 0.0)) for word in words]
        mean_word_conf = float(np.mean(observed_conf)) if observed_conf else 0.0
        structure_bonus = 0.08 if len(table_data.headers) > 1 and len(rows) > 1 else 0.0
        confidence = min(0.95, max(self.confidence, mean_word_conf * 0.85 + structure_bonus))
        return TableExtractionResult(
            region_id=region.region_id,
            table_data=table_data,
            confidence=confidence,
            format="tesseract_grid",
        )


class DummyTableExtractor(SimpleTableExtractor):
    """Backward-compatible alias for the heuristic table extractor."""
