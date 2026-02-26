from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import Counter
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


def _guess_content_type(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return "application/pdf"
    if suffix in {".png"}:
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix in {".tif", ".tiff"}:
        return "image/tiff"
    return "application/octet-stream"


def _default_bench_png() -> tuple[bytes, str, str]:
    import io

    from PIL import Image, ImageDraw

    img = Image.new("RGB", (512, 256), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((12, 12), "ZOCR API BENCH 123,456", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), "bench.png", "image/png"


async def _bench_api_run_async(
    *,
    url: str,
    endpoint: str,
    requests_total: int,
    concurrency: int,
    warmup: int,
    timeout_sec: float,
    api_key: Optional[str],
    toy_lite: bool,
    domain: Optional[str],
    dpi: int,
    k: int,
    seed: int,
    snapshot: bool,
    file_path: Optional[str],
) -> Dict[str, Any]:
    import asyncio
    import random

    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError(
            "httpx is required for API benchmarking. Install with `pip install -e '.[api]'` "
            "(or `pip install 'zocr-suite[api]'`)."
        ) from exc

    endpoint = endpoint.strip() or "/v1/run"
    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint
    base = (url or "").rstrip("/")
    if not base:
        base = "http://127.0.0.1:8000"
    target = base + endpoint

    if file_path:
        p = Path(file_path)
        payload = p.read_bytes()
        filename = p.name or "bench"
        content_type = _guess_content_type(filename)
    else:
        payload, filename, content_type = _default_bench_png()

    params: dict[str, Any] = {
        "domain": domain,
        "dpi": int(dpi),
        "k": int(k),
        "seed": int(seed),
        "snapshot": bool(snapshot),
        "toy_lite": bool(toy_lite),
    }
    params = {k: v for k, v in params.items() if v is not None}

    headers: dict[str, str] = {}
    if api_key:
        headers["X-API-Key"] = api_key

    requests_total = int(max(1, requests_total))
    concurrency = int(max(1, concurrency))
    warmup = int(max(0, warmup))
    timeout_sec = float(timeout_sec) if timeout_sec and timeout_sec > 0 else 300.0

    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    timeout = httpx.Timeout(timeout_sec)
    statuses: Counter[str] = Counter()
    exceptions: Counter[str] = Counter()
    lat: List[float] = []

    def _files():
        return {"file": (filename, payload, content_type)}

    async with httpx.AsyncClient(timeout=timeout, limits=limits, headers=headers) as client:
        for _ in range(warmup):
            try:
                res = await client.post(target, params=params, files=_files())
            except Exception as exc:
                _ = exc
            else:
                try:
                    await res.aread()
                except Exception:
                    pass

        statuses.clear()
        exceptions.clear()
        lat.clear()

        q: asyncio.Queue[int] = asyncio.Queue()
        for i in range(requests_total):
            q.put_nowait(i)

        async def worker(worker_id: int) -> None:
            rnd = random.Random(worker_id + 0xA11)
            while True:
                try:
                    _ = q.get_nowait()
                except asyncio.QueueEmpty:
                    break
                t0 = time.perf_counter()
                try:
                    res = await client.post(target, params=params, files=_files())
                except Exception as exc:
                    exceptions[type(exc).__name__] += 1
                else:
                    statuses[str(res.status_code)] += 1
                    # Best-effort consume the body to keep the connection reusable.
                    try:
                        if res.headers.get("content-type", "").startswith("application/json"):
                            _ = res.json()
                        else:
                            _ = res.content
                    except Exception:
                        pass
                    # Add a small jitter to spread out bursts when benchmarking locally.
                    if rnd.random() < 0.01:
                        await asyncio.sleep(0)
                lat.append(time.perf_counter() - t0)
                q.task_done()

        t_load0 = time.perf_counter()
        workers = [asyncio.create_task(worker(i)) for i in range(concurrency)]
        await asyncio.gather(*workers)
        load_sec = time.perf_counter() - t_load0

    success_2xx = sum(v for k, v in statuses.items() if k.startswith("2"))
    non_2xx = sum(v for k, v in statuses.items() if not k.startswith("2"))
    errors = int(sum(exceptions.values()))

    return {
        "name": "api.v1_run",
        "target": target,
        "params": params,
        "requests": requests_total,
        "concurrency": concurrency,
        "warmup": warmup,
        "timeout_sec": float(timeout_sec),
        "results": {
            "total_sec": float(load_sec),
            "rps": float(requests_total / load_sec) if load_sec > 0 else None,
            "success_2xx": int(success_2xx),
            "non_2xx": int(non_2xx),
            "exceptions": dict(exceptions),
            "http_status": dict(statuses),
            "latency_sec": {
                "min": min(lat) if lat else 0.0,
                "p50": _percentile(lat, 50.0),
                "p90": _percentile(lat, 90.0),
                "p95": _percentile(lat, 95.0),
                "p99": _percentile(lat, 99.0),
                "mean": mean(lat) if lat else 0.0,
                "max": max(lat) if lat else 0.0,
            },
            "error_total": int(errors),
        },
    }


def _bench_api_run(**kwargs) -> Dict[str, Any]:
    import asyncio

    return asyncio.run(_bench_api_run_async(**kwargs))


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser("zocr bench")
    sub = parser.add_subparsers(dest="bench")

    toy = sub.add_parser("toy", help="Benchmark toy OCR export loop.")
    toy.add_argument("--iterations", type=int, default=5)
    toy.add_argument("--warmup", type=int, default=1)
    toy.add_argument("--out", type=str, default=None, help="Optional JSON output path.")

    api = sub.add_parser("api", help="Benchmark the FastAPI /v1/run endpoint.")
    api.add_argument("--url", type=str, default="http://127.0.0.1:8000")
    api.add_argument("--endpoint", type=str, default="/v1/run")
    api.add_argument("--requests", type=int, default=20)
    api.add_argument("--concurrency", type=int, default=4)
    api.add_argument("--warmup", type=int, default=2)
    api.add_argument("--timeout-sec", type=float, default=300.0)
    api.add_argument("--api-key", type=str, default=None)
    api.add_argument("--toy-lite", action=argparse.BooleanOptionalAction, default=True)
    api.add_argument("--domain", type=str, default=None)
    api.add_argument("--dpi", type=int, default=200)
    api.add_argument("--k", type=int, default=10)
    api.add_argument("--seed", type=int, default=24601)
    api.add_argument("--snapshot", action=argparse.BooleanOptionalAction, default=False)
    api.add_argument("--file", type=str, default=None, help="Optional file to upload (pdf/png/jpg/tiff).")
    api.add_argument("--out", type=str, default=None, help="Optional JSON output path.")

    args = parser.parse_args(argv)

    bench_name = args.bench or "toy"
    if bench_name == "toy":
        bench = _bench_toy_export(int(args.iterations), int(args.warmup))
    elif bench_name == "api":
        bench = _bench_api_run(
            url=str(args.url),
            endpoint=str(args.endpoint),
            requests_total=int(args.requests),
            concurrency=int(args.concurrency),
            warmup=int(args.warmup),
            timeout_sec=float(args.timeout_sec),
            api_key=getattr(args, "api_key", None),
            toy_lite=bool(getattr(args, "toy_lite", True)),
            domain=getattr(args, "domain", None),
            dpi=int(getattr(args, "dpi", 200)),
            k=int(getattr(args, "k", 10)),
            seed=int(getattr(args, "seed", 24601)),
            snapshot=bool(getattr(args, "snapshot", False)),
            file_path=getattr(args, "file", None),
        )
    else:
        raise SystemExit("Unknown bench: " + bench_name)

    payload: Dict[str, Any] = {
        "schema": "zocr.bench",
        "schema_version": 1,
        "zocr_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "bench": bench,
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
