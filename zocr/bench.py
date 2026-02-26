from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from statistics import mean, median
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional

from ._version import __version__


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    if pct <= 0:
        return min(values)
    if pct >= 100:
        return max(values)
    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(len(ordered) - 1, lo + 1)
    if lo == hi:
        return ordered[lo]
    frac = k - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _bench_toy_export(iterations: int, warmup: int) -> Dict[str, Any]:
    from PIL import Image

    from zocr.consensus import toy_runtime

    x1, y1, x2, y2 = 10, 10, 210, 210
    col_bounds = [0, 100, 200]
    row_bands_rel = [(0, 40), (40, 80), (80, 120), (120, 160), (160, 200)]
    doc = {
        "doc_id": "bench",
        "pages": [
            {
                "index": 0,
                "tables": [
                    {
                        "bbox": [x1, y1, x2, y2],
                        "dbg": {"col_bounds": col_bounds, "row_bands_rel": row_bands_rel},
                    }
                ],
            }
        ],
    }

    with TemporaryDirectory(prefix="zocr_bench_") as tmp_dir:
        tmp = Path(tmp_dir)
        img_path = tmp / "page.png"
        Image.new("RGB", (220, 220), (255, 255, 255)).save(img_path)
        doc_path = tmp / "doc.zocr.json"
        doc_path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        out_path = tmp / "out.jsonl"

        for _ in range(max(0, warmup)):
            toy_runtime.export_jsonl_with_ocr(
                str(doc_path),
                str(img_path),
                str(out_path),
                ocr_engine="toy",
                contextual=False,
            )

        timings: List[float] = []
        records = None
        for _ in range(max(1, iterations)):
            t0 = time.perf_counter()
            records = toy_runtime.export_jsonl_with_ocr(
                str(doc_path),
                str(img_path),
                str(out_path),
                ocr_engine="toy",
                contextual=False,
            )
            timings.append(time.perf_counter() - t0)

    return {
        "name": "toy.export_jsonl_with_ocr",
        "iterations": int(max(1, iterations)),
        "warmup": int(max(0, warmup)),
        "records_last_run": records,
        "timings_sec": {
            "min": min(timings) if timings else 0.0,
            "p50": median(timings) if timings else 0.0,
            "p90": _percentile(timings, 90.0),
            "mean": mean(timings) if timings else 0.0,
            "max": max(timings) if timings else 0.0,
        },
    }


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser("zocr bench")
    sub = parser.add_subparsers(dest="bench")

    toy = sub.add_parser("toy", help="Benchmark toy OCR export loop.")
    toy.add_argument("--iterations", type=int, default=5)
    toy.add_argument("--warmup", type=int, default=1)
    toy.add_argument("--out", type=str, default=None, help="Optional JSON output path.")

    args = parser.parse_args(argv)

    bench_name = args.bench or "toy"
    if bench_name != "toy":
        raise SystemExit("Unknown bench: " + bench_name)

    payload: Dict[str, Any] = {
        "schema": "zocr.bench",
        "schema_version": 1,
        "zocr_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "bench": _bench_toy_export(int(args.iterations), int(args.warmup)),
    }

    out = getattr(args, "out", None)
    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()

