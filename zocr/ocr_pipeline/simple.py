"""Minimal built-in implementations for the OCR pipeline.

These classes are intentionally lightweight so the pipeline can run without
external ML models while still exercising the end-to-end flow on real images.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple, TypedDict

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


class OcrWord(TypedDict):
    text: str
    left: int
    top: int
    width: int
    height: int
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

    def segment(self, page: PageInput) -> List[SegmentedRegion]:
        image = page.image
        if not isinstance(image, Image.Image):
            raise TypeError("FullPageSegmenter expects a PIL.Image instance")

        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError("Page image must have positive dimensions")

        gray = _to_gray_array(image)
        mask = _ink_mask(gray)
        row_density = _smooth_density(mask.mean(axis=1), self.smoothing_window)

        min_row_gap = max(4, int(height * self.min_gap_fraction))
        min_col_gap = max(4, int(width * self.min_gap_fraction))

        row_gaps = _find_gaps(row_density, self.gap_ratio, min_row_gap)
        row_segments = _segments_from_gaps(height, row_gaps)

        regions: List[SegmentedRegion] = []
        min_region_size = max(8, int(min(width, height) * self.min_region_fraction))
        region_index = 0
        for row_start, row_end in row_segments:
            row_slice = mask[row_start:row_end, :]
            if row_slice.size == 0:
                continue
            col_density_slice = _smooth_density(
                row_slice.mean(axis=0),
                min(self.smoothing_window, max(3, row_slice.shape[1] // 50)),
            )
            col_gaps = _find_gaps(col_density_slice, self.gap_ratio, min_col_gap)
            col_segments = _segments_from_gaps(width, col_gaps)
            for col_start, col_end in col_segments:
                crop_mask = row_slice[:, col_start:col_end]
                if not crop_mask.any():
                    continue
                x0, y0, region_w, region_h = _bounding_box_from_mask(
                    crop_mask, col_start, row_start, min_region_size
                )
                x0, y0, region_w, region_h = _pad_box(
                    (x0, y0, region_w, region_h), self.bbox_padding, width, height
                )
                if region_w <= 0 or region_h <= 0:
                    continue
                if region_w < min_region_size and region_h < min_region_size:
                    continue
                region_index += 1
                region_id = f"{page.document_id}-page{page.page_number}-region{region_index}"
                bbox = BoundingBox(x=x0, y=y0, width=region_w, height=region_h)
                crop = image.crop((x0, y0, x0 + region_w, y0 + region_h))
                regions.append(
                    SegmentedRegion(
                        region_id=region_id,
                        bounding_box=bbox,
                        image_crop=crop,
                        confidence=self.confidence,
                        reading_order=region_index - 1,
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


class SimpleVLLM(VLLM):
    """Local visual descriptor for image-like regions.

    This is deliberately offline and deterministic. It does not claim semantic
    object recognition, but it produces richer cues than a bare placeholder so
    downstream review can distinguish photos, screenshots, diagrams, and flat
    graphics without adding a required model dependency.
    """

    def __init__(self, confidence: float = 0.55) -> None:
        self.confidence = confidence

    def describe(self, region: ClassifiedRegion) -> ImageCaptionResult:
        caption_parts: List[str] = []
        detected: List[str] = []
        detail: List[str] = []

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


class DummyVLLM(SimpleVLLM):
    """Backward-compatible alias for the heuristic VLM."""


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
            conf = data.get("conf", [])[idx] if idx < len(data.get("conf", [])) else -1
            try:
                conf_val = float(conf)
            except (TypeError, ValueError):
                conf_val = -1.0
            if conf_val < 0:
                continue
            left = data.get("left", [0])[idx] if idx < len(data.get("left", [])) else 0
            top = data.get("top", [0])[idx] if idx < len(data.get("top", [])) else 0
            width = data.get("width", [0])[idx] if idx < len(data.get("width", [])) else 0
            height = data.get("height", [0])[idx] if idx < len(data.get("height", [])) else 0
            block_num = data.get("block_num", [0])[idx] if idx < len(data.get("block_num", [])) else 0
            par_num = data.get("par_num", [0])[idx] if idx < len(data.get("par_num", [])) else 0
            line_num = data.get("line_num", [0])[idx] if idx < len(data.get("line_num", [])) else 0
            words.append(
                {
                    "text": text.strip(),
                    "left": int(left),
                    "top": int(top),
                    "width": int(width),
                    "height": int(height),
                    "row_key": (int(block_num), int(par_num), int(line_num)),
                }
            )

        if horizontal and vertical:
            grid_rows = _assign_words_to_grid(words, horizontal, vertical)
            table_data = _table_data_from_rows(grid_rows, len(vertical) - 1)
            confidence = min(
                0.92,
                self.confidence + 0.12 + (0.03 * min(table_data.num_rows or 0, 4)),
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

        words.sort(key=lambda w: (w["row_key"], w["top"], w["left"]))
        word_widths = [w["width"] for w in words if w["width"] > 0]
        median_width = float(np.median(word_widths)) if word_widths else 20.0
        cluster_gap = max(12.0, median_width * 1.6)
        centers = _cluster_centers((w["left"] + w["width"] / 2 for w in words), cluster_gap)
        centers = sorted(centers)
        if not centers:
            centers = [0.0]

        line_values = sorted({word["row_key"] for word in words})
        use_line_numbers = len(line_values) > 1 and any(
            any(value > 0 for value in key) for key in line_values
        )

        lines: Dict[LineKey, List[OcrWord]] = {}
        if use_line_numbers:
            for word in words:
                lines.setdefault(word["row_key"], []).append(word)
        else:
            heights = [w["height"] for w in words if w["height"] > 0]
            median_height = float(np.median(heights)) if heights else 10.0
            row_gap = max(8.0, median_height * 1.4)
            row_centers = _cluster_centers(
                (w["top"] + w["height"] / 2 for w in words),
                row_gap,
            )
            row_centers = sorted(row_centers)
            if not row_centers:
                row_centers = [0.0]
            for word in words:
                center = word["top"] + word["height"] / 2
                row_idx = min(
                    range(len(row_centers)), key=lambda i: abs(row_centers[i] - center)
                )
                lines.setdefault((row_idx, 0, 0), []).append(word)

        ordered_lines = [lines[key] for key in sorted(lines.keys())]
        rows: List[List[str]] = []
        for line_words in ordered_lines:
            cells = ["" for _ in centers]
            for word in line_words:
                center = word["left"] + word["width"] / 2
                column_idx = min(range(len(centers)), key=lambda i: abs(centers[i] - center))
                cells[column_idx] = (cells[column_idx] + " " + word["text"]).strip()
            rows.append(cells)

        table_data = _table_data_from_rows(rows, len(centers))
        confidence = min(0.9, self.confidence + (0.05 * min(table_data.num_rows or 0, 4)))
        return TableExtractionResult(
            region_id=region.region_id,
            table_data=table_data,
            confidence=confidence,
            format="tesseract",
        )


class DummyTableExtractor(SimpleTableExtractor):
    """Backward-compatible alias for the heuristic table extractor."""
