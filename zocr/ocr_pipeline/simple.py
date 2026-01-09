"""Minimal built-in implementations for the OCR pipeline.

These classes are intentionally lightweight so the pipeline can run without
external ML models while still exercising the end-to-end flow on real images.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pytesseract
from PIL import Image

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


def _to_gray_array(image: Image.Image) -> np.ndarray:
    gray = image.convert("L")
    return np.array(gray)


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    if gray.size == 0:
        return np.zeros_like(gray, dtype=bool)
    percentile = float(np.percentile(gray, 70))
    threshold = max(0.0, min(255.0, percentile - 12.0))
    return gray < threshold


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


def _box_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    ax1, ay1 = box_a.x, box_a.y
    ax2, ay2 = box_a.x + box_a.width, box_a.y + box_a.height
    bx1, by1 = box_b.x, box_b.y
    bx2, by2 = box_b.x + box_b.width, box_b.y + box_b.height

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = box_a.width * box_a.height
    area_b = box_b.width * box_b.height
    union = max(1.0, area_a + area_b - inter_area)
    return float(inter_area / union)


def _boxes_adjacent(box_a: BoundingBox, box_b: BoundingBox, gap: int) -> bool:
    ax1, ay1 = box_a.x, box_a.y
    ax2, ay2 = box_a.x + box_a.width, box_a.y + box_a.height
    bx1, by1 = box_b.x, box_b.y
    bx2, by2 = box_b.x + box_b.width, box_b.y + box_b.height
    horizontal_gap = max(bx1 - ax2, ax1 - bx2, 0)
    vertical_gap = max(by1 - ay2, ay1 - by2, 0)
    return horizontal_gap <= gap and vertical_gap <= gap


def _merge_regions(
    regions: Sequence[SegmentedRegion],
    image: Image.Image,
    overlap_threshold: float,
    adjacency_gap: int,
    confidence: float,
) -> List[SegmentedRegion]:
    merged: List[SegmentedRegion] = []
    for region in regions:
        merged_into = None
        for existing in merged:
            iou = _box_iou(existing.bounding_box, region.bounding_box)
            if iou >= overlap_threshold or _boxes_adjacent(
                existing.bounding_box, region.bounding_box, adjacency_gap
            ):
                merged_into = existing
                break
        if merged_into is None:
            merged.append(region)
            continue

        x1 = min(merged_into.bounding_box.x, region.bounding_box.x)
        y1 = min(merged_into.bounding_box.y, region.bounding_box.y)
        x2 = max(
            merged_into.bounding_box.x + merged_into.bounding_box.width,
            region.bounding_box.x + region.bounding_box.width,
        )
        y2 = max(
            merged_into.bounding_box.y + merged_into.bounding_box.height,
            region.bounding_box.y + region.bounding_box.height,
        )
        x2 = min(x2, image.width)
        y2 = min(y2, image.height)
        merged_into.bounding_box = BoundingBox(
            x=x1, y=y1, width=max(1, x2 - x1), height=max(1, y2 - y1)
        )
        merged_into.image_crop = image.crop((x1, y1, x2, y2))
        merged_into.confidence = max(merged_into.confidence, region.confidence, confidence)

    merged_sorted = sorted(
        merged,
        key=lambda r: (
            r.bounding_box.y,
            r.bounding_box.x,
            r.bounding_box.height,
            r.bounding_box.width,
        ),
    )
    output: List[SegmentedRegion] = []
    for idx, region in enumerate(merged_sorted, start=1):
        region.reading_order = idx - 1
        region.region_id = f"{region.region_id.rsplit('-', 1)[0]}-region{idx}"
        output.append(region)
    return output


class FullPageSegmenter(Segmenter):
    """Split the page into multiple regions using whitespace projections."""

    def __init__(
        self,
        confidence: float = 0.85,
        gap_ratio: float = 0.015,
        min_gap_fraction: float = 0.02,
        min_region_fraction: float = 0.08,
        merge_overlap_threshold: float = 0.2,
        merge_adjacency_gap: int = 6,
    ) -> None:
        self.confidence = confidence
        self.gap_ratio = gap_ratio
        self.min_gap_fraction = min_gap_fraction
        self.min_region_fraction = min_region_fraction
        self.merge_overlap_threshold = merge_overlap_threshold
        self.merge_adjacency_gap = merge_adjacency_gap

    def segment(self, page: PageInput) -> List[SegmentedRegion]:
        image = page.image
        if not isinstance(image, Image.Image):
            raise TypeError("FullPageSegmenter expects a PIL.Image instance")

        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError("Page image must have positive dimensions")

        gray = _to_gray_array(image)
        mask = _ink_mask(gray)
        row_density = mask.mean(axis=1)
        col_density = mask.mean(axis=0)

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
            col_density_slice = row_slice.mean(axis=0)
            col_gaps = _find_gaps(col_density_slice, self.gap_ratio, min_col_gap)
            col_segments = _segments_from_gaps(width, col_gaps)
            for col_start, col_end in col_segments:
                crop_mask = row_slice[:, col_start:col_end]
                if not crop_mask.any():
                    continue
                x0, y0, region_w, region_h = _bounding_box_from_mask(
                    crop_mask, col_start, row_start, min_region_size
                )
                if region_w <= 0 or region_h <= 0:
                    continue
                if region_w < min_region_size and region_h < min_region_size:
                    continue
                x1 = min(width, x0 + region_w)
                y1 = min(height, y0 + region_h)
                region_w = max(1, x1 - x0)
                region_h = max(1, y1 - y0)
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

        if len(regions) > 1:
            regions = _merge_regions(
                regions,
                image,
                overlap_threshold=self.merge_overlap_threshold,
                adjacency_gap=self.merge_adjacency_gap,
                confidence=self.confidence,
            )
        else:
            regions[0].reading_order = 0

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

            if row_ratio > 0.03 and col_ratio > 0.02 and ink_ratio > 0.01:
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


class SimpleVLLM(VLLM):
    """Heuristic captioner for image-like regions."""

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
            variance = float(np.var(gray)) if gray.size else 0.0
            dominant = _dominant_color(image)

            caption_parts.append(f"Image region {width}x{height}")
            detail.append(f"dominant color {dominant}")
            detail.append(f"edge density {edge_ratio:.2f}")

            if ink_ratio < 0.02 and edge_ratio > 0.07:
                caption_parts.append("photo-like content")
                detected.append("photo")
            elif ink_ratio > 0.05 and edge_ratio > 0.05:
                caption_parts.append("diagram or chart")
                detected.append("diagram")
            elif variance < 120:
                caption_parts.append("flat graphic")
                detected.append("graphic")
            else:
                caption_parts.append("mixed visual content")
            detail.append(f"ink ratio {ink_ratio:.3f}")
            detail.append(f"variance {variance:.1f}")
        else:
            caption_parts.append(f"Image region {region.region_id}")

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


class SimpleTableExtractor(TableExtractor):
    """Extract tables using pytesseract word boxes and column clustering."""

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

        try:
            data = pytesseract.image_to_data(
                region.image_crop, output_type=pytesseract.Output.DICT
            )
        except (pytesseract.TesseractNotFoundError, RuntimeError, ValueError):
            table_data = TableData(headers=["col1"], rows=[], num_rows=0, num_columns=1)
            return TableExtractionResult(
                region_id=region.region_id,
                table_data=table_data,
                confidence=0.2,
                format="unavailable",
            )
        words = []
        for text, conf, left, top, width, height, line_num in zip(
            data.get("text", []),
            data.get("conf", []),
            data.get("left", []),
            data.get("top", []),
            data.get("width", []),
            data.get("height", []),
            data.get("line_num", []),
        ):
            if not text or text.strip() == "":
                continue
            try:
                conf_val = float(conf)
            except (TypeError, ValueError):
                conf_val = -1.0
            if conf_val < 0:
                continue
            words.append(
                {
                    "text": text.strip(),
                    "left": int(left),
                    "top": int(top),
                    "width": int(width),
                    "height": int(height),
                    "line": int(line_num),
                }
            )

        if not words:
            table_data = TableData(headers=["col1"], rows=[], num_rows=0, num_columns=1)
            return TableExtractionResult(
                region_id=region.region_id,
                table_data=table_data,
                confidence=0.25,
                format="empty",
            )

        words.sort(key=lambda w: (w["line"], w["top"], w["left"]))
        word_widths = [w["width"] for w in words if w["width"] > 0]
        median_width = float(np.median(word_widths)) if word_widths else 20.0
        cluster_gap = max(12.0, median_width * 1.6)
        centers = _cluster_centers((w["left"] + w["width"] / 2 for w in words), cluster_gap)
        centers = sorted(centers)
        if not centers:
            centers = [0.0]

        lines: Dict[int, List[dict]] = {}
        for word in words:
            lines.setdefault(word["line"], []).append(word)

        ordered_lines = [lines[key] for key in sorted(lines.keys())]
        rows: List[List[str]] = []
        for line_words in ordered_lines:
            cells = ["" for _ in centers]
            for word in line_words:
                center = word["left"] + word["width"] / 2
                column_idx = min(range(len(centers)), key=lambda i: abs(centers[i] - center))
                cells[column_idx] = (cells[column_idx] + " " + word["text"]).strip()
            rows.append(cells)

        header_cells: List[str] = []
        data_rows = rows
        if rows and _is_header_candidate(rows[0]):
            header_cells = [cell if cell else f"col{idx+1}" for idx, cell in enumerate(rows[0])]
            data_rows = rows[1:]
        else:
            header_cells = [f"col{idx+1}" for idx in range(len(centers))]

        row_dicts = [
            {header: row[idx] if idx < len(row) else "" for idx, header in enumerate(header_cells)}
            for row in data_rows
        ]

        table_data = TableData(
            headers=header_cells,
            rows=row_dicts,
            num_rows=len(row_dicts),
            num_columns=len(header_cells),
        )
        confidence = min(0.9, self.confidence + (0.05 * min(len(row_dicts), 4)))
        return TableExtractionResult(
            region_id=region.region_id,
            table_data=table_data,
            confidence=confidence,
            format="tesseract",
        )


class DummyTableExtractor(SimpleTableExtractor):
    """Backward-compatible alias for the heuristic table extractor."""
