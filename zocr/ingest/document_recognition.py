from __future__ import annotations

import os
from pathlib import Path

from PIL import Image

from ..artifacts.evidence import archive_page_image
from ..artifacts.manifest import _sha256_file
from ..consensus.ocr_quality import assess_recognition


def prepare_image_pages(page_images, native_pages, outdir, *, recognizer, source_documents=None,
                        requested_language="ja") -> dict:
    plans = dict(native_pages)
    for index, image_path in page_images.items():
        native = plans.get(index, {})
        if native.get("route") == "native":
            continue
        path = Path(image_path)
        image_digest = _sha256_file(path)
        source = (source_documents or {}).get(os.path.abspath(image_path), {})
        digest = source.get("sha256", native.get("source_sha256", image_digest))
        prefix = "pdf" if source or native.get("records") else "image"
        with Image.open(path) as image:
            archived = archive_page_image(image, outdir)
            if archived is None:
                raise ValueError("OCR page evidence exceeds the archive budget")
            regions = native.get("ocr_regions") or [[0, 0, image.width, image.height]]
            records = list(native.get("records", []))
            for region in regions:
                x0, y0, x1, y1 = region
                if not 0 <= x0 < x1 <= image.width or not 0 <= y0 < y1 <= image.height:
                    raise ValueError("OCR region exceeds its source page")
                response = recognizer.recognize(image.crop(region))
                if response.get("engine") != "apple_vision":
                    raise ValueError("Unexpected document OCR provider")
                provider = {key: response.get(key) for key in (
                    "engine", "revision", "languages", "os_version", "provider_source_sha256",
                    "model_identity", "model_sha256", "language_correction", "device_policy", "build",
                )}
                lines = response["lines"] or [{"text": "", "confidence": 0.0,
                                               "bbox": [0, 0, x1 - x0, y1 - y0]}]
                for line in lines:
                    quality = assess_recognition(line["text"], line["confidence"],
                                                 requested_language=requested_language,
                                                 supported_languages=response["languages"])
                    left, top, right, bottom = line["bbox"]
                    if not 0 <= left < right <= x1 - x0 or not 0 <= top < bottom <= y1 - y0:
                        raise ValueError("OCR observation exceeds its source region")
                    records.append({"type": "text_region", "doc_id": f"{prefix}-{digest}", "page": index,
                                    "table_index": None, "row": None, "col": None,
                                    "bbox": [x0 + left, y0 + top, x0 + right, y0 + bottom],
                                    "image_path": archived, "text": quality["text"],
                                    "meta": {"source_kind": "image_ocr", "source_sha256": digest,
                                             "source_page": source.get("source_page", 0) + 1,
                                             "page_image_sha256": Path(archived).stem,
                                             "evidence_kind": "observed", "synthetic": False,
                                             "observed_text": quality["raw_text"], "record_kind": "text_region", "region_index": len(records),
                                             "confidence": {"ocr": quality["confidence"]}, "needs_review": True,
                                             "recognition_status": quality["status"],
                                             "abstention_reasons": quality["reasons"],
                                             "confidence_calibration": quality["confidence_calibration"],
                                             "ocr_normalized_bbox": line.get("normalized_bbox"),
                                             "ocr_bbox_clipped": line.get("bbox_clipped", False),
                                             "ocr_provider": provider}})
                    if len(records) > 10000:
                        raise ValueError("Document OCR exceeds the region budget")
            if _sha256_file(path) != image_digest:
                raise ValueError("Source bitmap changed during OCR")
        plans[index] = {"route": "mixed_vision" if native.get("records") else "vision",
                        "reason": "unverified_image_ocr", "records": records, "ocr_regions": regions,
                        "source_sha256": digest}
    return plans
