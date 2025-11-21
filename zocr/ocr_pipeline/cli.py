# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024 ZOCR contributors

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
from typing import Iterable, List, Sequence

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


def _load_images(paths: Iterable[str]) -> List[Image.Image]:
    return [Image.open(Path(p).as_posix()) for p in paths]


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

    if not args.images and not args.pdf:
        raise SystemExit("Provide --images or --pdf")

    pipeline = build_document_pipeline(use_mocks=args.use_mocks)

    if args.images:
        images = _load_images(args.images)
        document = DocumentInput(document_id=args.document_id, images=images)
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
