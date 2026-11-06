from __future__ import annotations

import math
from pathlib import Path
import unicodedata

from ..artifacts.evidence import archive_page_image
from ..artifacts.manifest import _sha256_file


def extract_native_page(pdf_path, source_page: int, image, outdir, *, page_number: int = 0,
                        expected_sha256: str | None = None) -> dict:
    if isinstance(source_page, bool) or not isinstance(source_page, int) or source_page < 0:
        raise ValueError("Invalid PDF source page")
    if isinstance(page_number, bool) or not isinstance(page_number, int) or page_number < 0:
        raise ValueError("Invalid output page")
    root = Path(outdir).resolve()
    path = Path(pdf_path)
    digest = _sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("PDF source no longer matches its rendered page")
    result = {"route": "raster", "reason": "no_usable_text", "records": [],
              "ocr_regions": [[0, 0, image.width, image.height]], "source_sha256": digest}
    try:
        import fitz
    except ImportError:
        return {**result, "reason": "native_parser_unavailable"}
    with fitz.open(path) as document:
        page = document[source_page]
        width, height = page.rect.width, page.rect.height
        if width <= 0 or height <= 0 or image.width <= 0 or image.height <= 0:
            raise ValueError("Invalid page dimensions")
        if abs((image.width / image.height) / (width / height) - 1) > .01:
            raise ValueError("Page bitmap aspect ratio does not match its PDF source")
        traces = page.get_texttrace()
        if any(span.get("type") == 3 or span.get("opacity", 1) < .01 for span in traces):
            return {**result, "reason": "hidden_text_layer"}

        def pixels(box):
            if len(box) != 4 or not all(math.isfinite(value) for value in box):
                raise ValueError("Invalid PDF text coordinates")
            rotated = fitz.Rect(box) * page.rotation_matrix
            mapped = [max(0, min(image.width, math.floor(rotated.x0 * image.width / width))),
                      max(0, min(image.height, math.floor(rotated.y0 * image.height / height))),
                      max(0, min(image.width, math.ceil(rotated.x1 * image.width / width))),
                      max(0, min(image.height, math.ceil(rotated.y1 * image.height / height)))]
            if mapped[0] >= mapped[2] or mapped[1] >= mapped[3]:
                return None
            return mapped

        flags = fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES
        blocks = page.get_text("dict", flags=flags)["blocks"]
        lines = [line for block in blocks for line in block.get("lines", [])]
        if len(lines) > 10000:
            raise ValueError("PDF native text exceeds the region budget")
        candidates = []
        characters = 0
        for line in lines:
            text = "".join(span["text"] for span in line["spans"]).strip()
            if not text:
                continue
            characters += len(text)
            if characters > 1_000_000:
                raise ValueError("PDF native text exceeds the character budget")
            if any(char == "\ufffd" or unicodedata.category(char) in {"Co", "Cs", "Cc"}
                   for char in text if char not in "\t\n\r"):
                return {**result, "reason": "unusable_text_layer"}
            box = pixels(line["bbox"])
            if box is None:
                return {**result, "reason": "unusable_text_coordinates"}
            candidates.append((text, box, list(line["bbox"])))
        if not candidates:
            return result
        regions = []
        for info in page.get_image_info():
            box = pixels(info["bbox"])
            if box and box not in regions:
                regions.append(box)
        archived = archive_page_image(image, root)
        if archived is None:
            raise ValueError("Native PDF page evidence exceeds the archive budget")
        if _sha256_file(path) != digest:
            raise ValueError("PDF source changed during native extraction")
        relative_image = Path(archived).relative_to(root).as_posix()
        records = [{"doc_id": f"pdf-{digest}", "page": page_number, "table_index": None,
                    "row": None, "col": None, "bbox": box, "text": text,
                    "image_path": relative_image,
                    "meta": {"source_kind": "pdf_text_layer", "source_sha256": digest,
                             "source_page": source_page + 1, "pdf_bbox": pdf_box,
                             "page_rotation": page.rotation, "page_image_sha256": Path(archived).stem,
                             "evidence_kind": "observed", "observed_text": text, "synthetic": False,
                             "record_kind": "text_region", "confidence": {}, "needs_review": True,
                             "visual_verification": "not_performed", "native_order": index}}
                   for index, (text, box, pdf_box) in enumerate(candidates)]
    return {"route": "mixed" if regions else "native", "records": records, "ocr_regions": regions,
            "source_sha256": digest, "reason": "unverified_pdf_text_layer"}


def mark_unrecognized_images(plan: dict) -> dict:
    records = list(plan["records"])
    template = records[0]
    source = template["meta"]
    for region in plan["ocr_regions"]:
        records.append({**template, "text": "", "bbox": region,
                        "meta": {key: source[key] for key in (
                            "source_sha256", "source_page", "page_image_sha256", "record_kind",
                        )} | {"source_kind": "image_region", "evidence_kind": "observed", "observed_text": "",
                             "synthetic": False, "confidence": {}, "needs_review": True,
                             "recognition_status": "abstained", "required_ocr_missing": True,
                             "abstention_reasons": ["image_ocr_provider_not_selected"], "region_index": len(records)}})
    return {**plan, "route": "mixed_unrecognized", "records": records}
