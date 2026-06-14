"""Command-line entry for the lightweight OCR pipeline.

This CLI is intended for quick local runs or EC2 workflows where the
full orchestrator is unnecessary. It wires together the modular pipeline
components (either the built-in "simple" stack or mocks) and emits JSON
for each processed page.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

from PIL import Image

from . import (
    AspectRatioRegionClassifier,
    BasicInputHandler,
    DocumentInput,
    DocumentPipeline,
    DummyTableExtractor,
    FullPageSegmenter,
    HttpVLLM,
    MockAggregator,
    MockInputHandler,
    MockRegionClassifier,
    MockSegmenter,
    MockTableExtractor,
    MockTextOCR,
    MockVLLM,
    SimpleAggregator,
    SimpleVisualDescriptor,
    TesseractTextOCR,
    TwoStageTextOCR,
    ZocrRuntimeOCR,
)
from .interfaces import TextOCR
from .pipeline import OcrPipeline


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"}
_RUN_SCHEMA = "zocr.simple_run.v1"


def _load_image(path: Path) -> Image.Image:
    with Image.open(path.as_posix()) as image:
        return image.copy()


def _load_images(paths: Iterable[str]) -> List[Image.Image]:
    return [_load_image(Path(p)) for p in paths]


def _collect_images(directory: Path, pattern: str, recursive: bool) -> List[Path]:
    if not directory.exists():
        raise SystemExit(f"Input directory not found: {directory}")
    iterator = directory.rglob(pattern) if recursive else directory.glob(pattern)
    images = [path for path in iterator if path.suffix.lower() in _IMAGE_SUFFIXES]
    return sorted(images)


def _load_directory_images(directory: Path, pattern: str, recursive: bool) -> List[Image.Image]:
    images = _collect_images(directory, pattern, recursive)
    if not images:
        raise SystemExit(f"No images found in {directory} (pattern={pattern})")
    return _load_images([path.as_posix() for path in images])


def _load_batch_documents(
    directory: Path, pattern: str, recursive: bool
) -> List[Tuple[str, List[Image.Image]]]:
    if not directory.exists():
        raise SystemExit(f"Batch directory not found: {directory}")
    documents: List[Tuple[str, List[Image.Image]]] = []
    for subdir in sorted([path for path in directory.iterdir() if path.is_dir()]):
        images = _collect_images(subdir, pattern, recursive)
        if not images:
            continue
        documents.append((subdir.name, _load_images([path.as_posix() for path in images])))
    if not documents:
        raise SystemExit(f"No documents found under {directory}")
    return documents


def _build_default_text_ocr() -> TextOCR:
    primary = ZocrRuntimeOCR()
    try:
        fallback = TesseractTextOCR()
    except RuntimeError:
        return primary
    return TwoStageTextOCR(
        primary=primary,
        fallback=fallback,
        compare_primary_engines=("zocr_runtime", "toy_runtime"),
    )


def build_document_pipeline(
    *,
    use_mocks: bool = False,
    vllm_endpoint: str | None = None,
) -> DocumentPipeline:
    segmenter = MockSegmenter() if use_mocks else FullPageSegmenter()
    classifier = MockRegionClassifier() if use_mocks else AspectRatioRegionClassifier()
    text_ocr = MockTextOCR() if use_mocks else _build_default_text_ocr()
    endpoint = vllm_endpoint or os.environ.get("ZOCR_VLLM_ENDPOINT")
    vllm = MockVLLM() if use_mocks else (HttpVLLM(endpoint) if endpoint else SimpleVisualDescriptor())
    table_extractor = MockTableExtractor() if use_mocks else DummyTableExtractor()
    aggregator = MockAggregator() if use_mocks else SimpleAggregator()

    page_pipeline = OcrPipeline(
        segmenter=segmenter,
        region_classifier=classifier,
        text_ocr=text_ocr,
        vllm=vllm,
        table_extractor=table_extractor,
        aggregator=aggregator,
    )

    input_handler = MockInputHandler() if use_mocks else BasicInputHandler()

    return DocumentPipeline(input_handler=input_handler, page_pipeline=page_pipeline)


def _json_dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _documents_from_payload(payload: Any) -> List[dict[str, Any]]:
    if not isinstance(payload, list):
        return []
    if payload and all(isinstance(item, dict) and "pages" in item for item in payload):
        documents: List[dict[str, Any]] = []
        for item in payload:
            pages = item.get("pages") if isinstance(item.get("pages"), list) else []
            documents.append(
                {
                    "document_id": str(item.get("document_id") or ""),
                    "pages": [page for page in pages if isinstance(page, dict)],
                }
            )
        return documents
    pages = [page for page in payload if isinstance(page, dict)]
    document_id = str(pages[0].get("document_id") or "") if pages else ""
    return [{"document_id": document_id, "pages": pages}]


def _content_preview(region: dict[str, Any]) -> str:
    content = region.get("content") if isinstance(region.get("content"), dict) else {}
    if "text" in content:
        return str(content.get("text") or "")[:160]
    if "caption" in content:
        return str(content.get("caption") or "")[:160]
    table_data = content.get("table_data")
    if isinstance(table_data, dict):
        rows = table_data.get("num_rows")
        columns = table_data.get("num_columns")
        return f"table rows={rows if rows is not None else '?'} cols={columns if columns is not None else '?'}"
    return ""


def _flatten_regions(payload: Any) -> List[dict[str, Any]]:
    flattened: List[dict[str, Any]] = []
    for document in _documents_from_payload(payload):
        for page in document["pages"]:
            for region in page.get("regions", []) or []:
                if not isinstance(region, dict):
                    continue
                content = region.get("content") if isinstance(region.get("content"), dict) else {}
                confidence = content.get("confidence")
                flattened.append(
                    {
                        "document_id": page.get("document_id") or document["document_id"],
                        "page_number": page.get("page_number"),
                        "region_id": region.get("region_id"),
                        "type": region.get("type"),
                        "reading_order": region.get("reading_order"),
                        "bounding_box": region.get("bounding_box"),
                        "confidence": confidence,
                        "preview": _content_preview(region),
                    }
                )
    return flattened


def _summarize_payload(
    payload: Any,
    *,
    input_mode: str,
    use_mocks: bool,
    vllm_endpoint: str | None,
) -> dict[str, Any]:
    documents = _documents_from_payload(payload)
    doc_summaries = []
    totals = {
        "documents": len(documents),
        "pages": 0,
        "regions": 0,
        "text_regions": 0,
        "image_regions": 0,
        "table_regions": 0,
    }
    for document in documents:
        doc_totals = {
            "document_id": document["document_id"],
            "pages": len(document["pages"]),
            "regions": 0,
            "text_regions": 0,
            "image_regions": 0,
            "table_regions": 0,
        }
        for page in document["pages"]:
            metadata = page.get("metadata") if isinstance(page.get("metadata"), dict) else {}
            doc_totals["regions"] += int(metadata.get("total_regions") or len(page.get("regions", []) or []))
            doc_totals["text_regions"] += int(metadata.get("text_regions") or 0)
            doc_totals["image_regions"] += int(metadata.get("image_regions") or 0)
            doc_totals["table_regions"] += int(metadata.get("table_regions") or 0)
        for key in ("pages", "regions", "text_regions", "image_regions", "table_regions"):
            totals[key] += int(doc_totals[key])
        doc_summaries.append(doc_totals)

    return {
        "schema": _RUN_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_mode": input_mode,
        "engine": "mock" if use_mocks else "zocr_runtime",
        "vllm": "http" if vllm_endpoint else "local",
        "totals": totals,
        "documents": doc_summaries,
    }


def _write_run_artifacts(
    outdir: Path,
    payload: Any,
    *,
    summary: dict[str, Any],
) -> dict[str, str]:
    outdir.mkdir(parents=True, exist_ok=True)
    pages_path = outdir / "pages.json"
    summary_path = outdir / "summary.json"
    regions_path = outdir / "regions.jsonl"
    manifest_path = outdir / "manifest.json"

    pages_path.write_text(_json_dump(payload) + "\n", encoding="utf-8")
    flattened = _flatten_regions(payload)
    regions_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in flattened),
        encoding="utf-8",
    )
    artifacts = {
        "pages_json": pages_path.as_posix(),
        "summary_json": summary_path.as_posix(),
        "regions_jsonl": regions_path.as_posix(),
        "manifest_json": manifest_path.as_posix(),
    }
    enriched_summary = dict(summary)
    enriched_summary["artifacts"] = artifacts
    summary_path.write_text(_json_dump(enriched_summary) + "\n", encoding="utf-8")
    manifest = {
        "schema": _RUN_SCHEMA,
        "created_at": summary["created_at"],
        "summary": enriched_summary["totals"],
        "documents": enriched_summary["documents"],
        "artifacts": artifacts,
    }
    manifest_path.write_text(_json_dump(manifest) + "\n", encoding="utf-8")
    return artifacts


def _print_handoff(
    summary: dict[str, Any],
    artifacts: dict[str, str] | None = None,
    *,
    stream: Any = None,
) -> None:
    target = stream or sys.stdout
    totals = summary.get("totals", {})
    print(
        "ZOCR simple run complete: "
        f"{totals.get('documents', 0)} docs, "
        f"{totals.get('pages', 0)} pages, "
        f"{totals.get('regions', 0)} regions "
        f"({totals.get('text_regions', 0)} text, "
        f"{totals.get('image_regions', 0)} image, "
        f"{totals.get('table_regions', 0)} table)",
        file=target,
    )
    if artifacts:
        print("Artifacts:", file=target)
        for label, path in artifacts.items():
            print(f"  {label}: {path}", file=target)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the lightweight OCR pipeline")
    parser.add_argument(
        "--images",
        nargs="*",
        help="One or more image files to process as a single document in order",
    )
    parser.add_argument("--pdf", help="PDF file to process")
    parser.add_argument(
        "--input-dir",
        help="Directory of images to process as a single document (sorted by name)",
    )
    parser.add_argument(
        "--batch-dir",
        help="Directory containing subdirectories, each treated as a document",
    )
    parser.add_argument(
        "--pattern",
        default="*",
        help="Glob pattern for images in --input-dir/--batch-dir (default: *)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan --input-dir/--batch-dir for images",
    )
    parser.add_argument("--document-id", default="doc", help="Identifier for the document")
    parser.add_argument(
        "--out",
        help="Output JSON path or '-' for stdout. Defaults to stdout unless --outdir is set.",
    )
    parser.add_argument(
        "--outdir",
        help="Write a handoff run folder with pages.json, summary.json, regions.jsonl, and manifest.json",
    )
    parser.add_argument(
        "--use-mocks",
        action="store_true",
        help="Use mock components (no external dependencies) for fast smoke tests",
    )
    parser.add_argument(
        "--vllm-endpoint",
        help="HTTP JSON endpoint for image-region captioning (falls back to ZOCR_VLLM_ENDPOINT)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the human-readable completion summary when using --outdir",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)

    selected_inputs = [
        bool(args.images),
        bool(args.pdf),
        bool(args.input_dir),
        bool(args.batch_dir),
    ]
    if not any(selected_inputs):
        raise SystemExit("Provide --images, --pdf, --input-dir, or --batch-dir")
    if sum(selected_inputs) > 1:
        raise SystemExit("Provide exactly one of --images, --pdf, --input-dir, or --batch-dir")

    resolved_vllm_endpoint = args.vllm_endpoint or os.environ.get("ZOCR_VLLM_ENDPOINT")
    pipeline = build_document_pipeline(use_mocks=args.use_mocks, vllm_endpoint=resolved_vllm_endpoint)

    payload = []
    input_mode = "images"
    if args.images:
        images = _load_images(args.images)
        document = DocumentInput(document_id=args.document_id, images=images)
        outputs = pipeline.process(document)
        payload = [out.model_dump() for out in outputs]
    elif args.input_dir:
        input_mode = "input_dir"
        images = _load_directory_images(
            Path(args.input_dir), args.pattern, args.recursive
        )
        document = DocumentInput(document_id=args.document_id, images=images)
        outputs = pipeline.process(document)
        payload = [out.model_dump() for out in outputs]
    elif args.batch_dir:
        input_mode = "batch_dir"
        batch_outputs = []
        for document_id, images in _load_batch_documents(
            Path(args.batch_dir), args.pattern, args.recursive
        ):
            document = DocumentInput(document_id=document_id, images=images)
            outputs = pipeline.process(document)
            batch_outputs.append(
                {"document_id": document_id, "pages": [out.model_dump() for out in outputs]}
            )
        payload = batch_outputs
    else:
        input_mode = "pdf"
        document = DocumentInput(document_id=args.document_id, file_path=args.pdf)
        outputs = pipeline.process(document)
        payload = [out.model_dump() for out in outputs]

    summary = _summarize_payload(
        payload,
        input_mode=input_mode,
        use_mocks=args.use_mocks,
        vllm_endpoint=resolved_vllm_endpoint,
    )
    artifacts = None
    if args.outdir:
        artifacts = _write_run_artifacts(Path(args.outdir), payload, summary=summary)

    if args.out == "-":
        print(_json_dump(payload))
        if args.outdir and not args.quiet:
            _print_handoff(summary, artifacts, stream=sys.stderr)
    elif args.out:
        path = Path(args.out)
        path.write_text(_json_dump(payload) + "\n", encoding="utf-8")
        if args.outdir and not args.quiet:
            _print_handoff(summary, artifacts)
    elif args.outdir:
        if not args.quiet:
            _print_handoff(summary, artifacts)
    else:
        print(_json_dump(payload))


if __name__ == "__main__":  # pragma: no cover
    main()
