"""Command-line entry for the lightweight OCR pipeline.

This CLI is intended for quick local runs or EC2 workflows where the
full orchestrator is unnecessary. It wires together the modular pipeline
components (either the built-in "simple" stack or mocks) and emits JSON
for each processed page.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from PIL import Image

from . import (
    AspectRatioRegionClassifier,
    BasicInputHandler,
    DocumentInput,
    DocumentPipeline,
    DummyTableExtractor,
    DummyVLLM,
    FullPageSegmenter,
    MockAggregator,
    MockInputHandler,
    MockRegionClassifier,
    MockSegmenter,
    MockTableExtractor,
    MockTextOCR,
    MockVLLM,
    SimpleAggregator,
    TesseractTextOCR,
    ToyRuntimeTextOCR,
    TwoStageTextOCR,
)
from .pipeline import OcrPipeline


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"}


def _load_images(paths: Iterable[str]) -> List[Image.Image]:
    return [Image.open(Path(p).as_posix()) for p in paths]


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


def build_document_pipeline(*, use_mocks: bool = False) -> DocumentPipeline:
    segmenter = MockSegmenter() if use_mocks else FullPageSegmenter()
    classifier = MockRegionClassifier() if use_mocks else AspectRatioRegionClassifier()
    text_ocr = (
        MockTextOCR()
        if use_mocks
        else TwoStageTextOCR(primary=ToyRuntimeTextOCR(), fallback=TesseractTextOCR())
    )
    vllm = MockVLLM() if use_mocks else DummyVLLM()
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
    parser.add_argument("--out", default="-", help="Output file path or '-' for stdout")
    parser.add_argument(
        "--use-mocks",
        action="store_true",
        help="Use mock components (no external dependencies) for fast smoke tests",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)

    if not any([args.images, args.pdf, args.input_dir, args.batch_dir]):
        raise SystemExit("Provide --images, --pdf, --input-dir, or --batch-dir")

    pipeline = build_document_pipeline(use_mocks=args.use_mocks)

    payload = []
    if args.images:
        images = _load_images(args.images)
        document = DocumentInput(document_id=args.document_id, images=images)
        outputs = pipeline.process(document)
        payload = [out.model_dump() for out in outputs]
    elif args.input_dir:
        images = _load_directory_images(
            Path(args.input_dir), args.pattern, args.recursive
        )
        document = DocumentInput(document_id=args.document_id, images=images)
        outputs = pipeline.process(document)
        payload = [out.model_dump() for out in outputs]
    elif args.batch_dir:
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
        document = DocumentInput(document_id=args.document_id, file_path=args.pdf)
        outputs = pipeline.process(document)
        payload = [out.model_dump() for out in outputs]

    if args.out == "-":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        path = Path(args.out)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    main()
