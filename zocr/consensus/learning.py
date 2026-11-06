from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
from statistics import median
import subprocess
import sys
import time
from typing import Any

from PIL import Image, ImageOps

from .._version import __version__
from ..bench import _percentile


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _read_bounded(path: Path, limit: int) -> bytes:
    try:
        with path.open("rb") as stream:
            data = stream.read(limit + 1)
        if len(data) > limit:
            raise ValueError(f"{path}: file exceeds {limit} bytes")
        return data
    except OSError as exc:
        raise ValueError(f"{path}: unreadable dataset/image") from exc


def _glyph_image(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as image:
        if max(image.size) > 512 or min(image.size) < 1:
            raise ValueError("Expected a glyph crop no larger than 512 x 512 pixels")
        gray = ImageOps.autocontrast(image.convert("L"))
    w, h = gray.size
    dark = median([gray.getpixel(point) for point in ((0, 0), (w-1, 0), (0, h-1), (w-1, h-1))]) >= 128
    mask = gray.point(lambda value: 255 if (value < 128) == dark else 0)
    box = mask.getbbox()
    if not box:
        raise ValueError("Blank glyph crops cannot train or evaluate visible characters")
    mask = mask.crop(box)
    if max(mask.size) > 128:
        raise ValueError("Cropped glyph exceeds 128 x 128 pixels")
    return mask


def load_glyph_dataset(dataset: str | Path, allowed_chars: str) -> dict[str, Any]:
    if not isinstance(allowed_chars, str) or not allowed_chars or len(allowed_chars) > 256:
        raise ValueError("An explicit alphabet of 1..256 characters is required")
    path = Path(dataset).resolve()
    raw = _read_bounded(path, 2 * 1024 * 1024)
    samples = []
    ids, pixels, sources = set(), set(), set()
    groups = {"train": set(), "holdout": set()}
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("Expected a JSON object")
            for field in ("id", "group_id", "reviewer", "reason", "image"):
                if not isinstance(record.get(field), str) or not record[field].strip() or len(record[field]) > 512:
                    raise ValueError(f"Missing or invalid {field}")
            label = record.get("text")
            if not isinstance(label, str) or len(label) != 1 or label.isspace() or label not in allowed_chars:
                raise ValueError("Each crop requires one visible character in the explicit alphabet")
            split = record.get("split")
            if not isinstance(split, str) or split not in groups or record.get("approved") is not True:
                raise ValueError("Only approved train/holdout annotations are accepted")
            image_path = (path.parent / record["image"]).resolve()
            if not image_path.is_relative_to(path.parent):
                raise ValueError("Image is outside the dataset directory")
            image_bytes = _read_bounded(image_path, 4 * 1024 * 1024)
            image = _glyph_image(image_bytes)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            png = buffer.getvalue()
            pixel_hash = hashlib.sha256(png).hexdigest()
            source_hash = hashlib.sha256(image_bytes).hexdigest()
            if record["id"] in ids or pixel_hash in pixels or source_hash in sources:
                raise ValueError("Duplicate IDs or repeated image pixels would contaminate the evaluation")
            ids.add(record["id"])
            pixels.add(pixel_hash)
            sources.add(source_hash)
            groups[split].add(record["group_id"])
            samples.append({"id": record["id"], "label": label, "split": split, "group_id": record["group_id"],
                            "png": base64.b64encode(png).decode("ascii"), "glyph_sha256": pixel_hash,
                            "source_sha256": source_hash, "annotation_sha256": hashlib.sha256(_json_bytes(record)).hexdigest()})
            if len(samples) > 1000:
                raise ValueError("At most 1000 glyph samples are supported per trial")
        except (ValueError, TypeError, OSError, Image.DecompressionBombError) as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not groups["train"] or len(groups["holdout"]) < 2:
        raise ValueError("Training data and holdout samples from at least two independent document groups are required")
    if groups["train"] & groups["holdout"]:
        raise ValueError("A document group cannot appear in both training and holdout splits")
    return {"dataset_sha256": hashlib.sha256(raw).hexdigest(), "allowed_chars": allowed_chars, "samples": samples}


def select_candidate(baseline: list[dict], candidate: list[dict]) -> dict[str, Any]:
    if not baseline or len(baseline) != len(candidate):
        raise ValueError("Candidate and baseline must evaluate the same nonempty holdout")
    regressions = sum(a["correct"] and not b["correct"] for a, b in zip(baseline, candidate))
    improvement = sum(row["correct"] for row in candidate) - sum(row["correct"] for row in baseline)
    return {"eligible": improvement > 0 and regressions == 0, "correct_delta": improvement, "regressions": regressions,
            "policy": "Strict holdout improvement without individual regressions; never auto-activate in production.",
            "requires_independent_test": True}


def _predict(runtime, samples: list[dict], alphabet: str) -> list[dict]:
    predictions = []
    for sample in samples:
        with Image.open(io.BytesIO(base64.b64decode(sample["png"], validate=True))) as image:
            if max(image.size) > 128:
                raise ValueError("Oversized worker glyph")
            started = time.perf_counter()
            text, confidence = runtime._match_glyph(image, runtime._GLYPH_ATLAS, allowed_chars=alphabet)
            elapsed = (time.perf_counter() - started) * 1000
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Recognizer returned an invalid confidence")
        predictions.append({"id": sample["id"], "expected": sample["label"], "predicted": text,
                            "correct": text == sample["label"], "confidence": confidence, "latency_ms": elapsed})
    return predictions


def _metrics(rows: list[dict]) -> dict[str, Any]:
    correct = sum(row["correct"] for row in rows)
    times = [row["latency_ms"] for row in rows]
    return {"samples": len(rows), "correct": correct, "accuracy": correct / len(rows),
            "glyph_error_rate": 1 - correct / len(rows), "median_ms": median(times),
            "p95_ms": _percentile(times, 95),
            "brier_score": sum((row["confidence"] - row["correct"]) ** 2 for row in rows) / len(rows),
            "high_confidence_errors": sum(not row["correct"] and row["confidence"] >= .9 for row in rows)}


def _worker(request: dict) -> dict[str, Any]:
    import numpy
    import PIL
    from . import toy_runtime as runtime

    baseline_model = runtime._toy_memory_payload()
    holdout = [sample for sample in request["samples"] if sample["split"] == "holdout"]
    _predict(runtime, holdout, request["allowed_chars"])
    baseline = _predict(runtime, holdout, request["allowed_chars"])
    trained = 0
    for sample in request["samples"]:
        if sample["split"] == "train":
            with Image.open(io.BytesIO(base64.b64decode(sample["png"], validate=True))) as image:
                trained += bool(runtime._learn_glyph_variant(sample["label"], image))
    memory = runtime._toy_memory_payload()
    _predict(runtime, holdout, request["allowed_chars"])
    candidate = _predict(runtime, holdout, request["allowed_chars"])
    selection = select_candidate(baseline, candidate)
    model_hash = hashlib.sha256(_json_bytes(baseline_model)).hexdigest()
    memory["learning_trial"] = {"dataset_sha256": request["dataset_sha256"], "parent_model_sha256": model_hash,
                                "training_glyph_sha256": [sample["glyph_sha256"] for sample in request["samples"]
                                                          if sample["split"] == "train"]}
    report = {"schema": "zocr.glyph_learning_trial.v1", "scope": "isolated_glyph_matching",
              "dataset_sha256": request["dataset_sha256"], "baseline_model_sha256": model_hash,
              "allowed_chars": request["allowed_chars"], "trained_variants": trained,
              "timing_policy": "One warmup pass per model; latency excludes crop loading/preprocessing.",
              "runtime_sha256": hashlib.sha256(Path(runtime.__file__).read_bytes()).hexdigest(),
              "learning_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "baseline": _metrics(baseline), "candidate": _metrics(candidate), "selection": selection,
              "predictions": {"baseline": baseline, "candidate": candidate},
              "samples": [{key: value for key, value in sample.items() if key != "png"} for sample in request["samples"]],
              "environment": {"zocr": __version__, "python": platform.python_version(), "pillow": PIL.__version__,
                              "numpy": numpy.__version__},
              "limitations": ["Holdout labels are selection data, not an independent final test.",
                              "Local approval metadata is asserted by the dataset owner, not server authenticated.",
                              "This does not measure page segmentation, full-text CER/WER or real-document accuracy."]}
    if selection["eligible"]:
        report["candidate_memory_sha256"] = hashlib.sha256(_json_bytes(memory)).hexdigest()
    return {"report": report, "memory": memory if selection["eligible"] else None}


def run_learning_trial(dataset: str | Path, outdir: str | Path, *, allowed_chars: str,
                       timeout_sec: float = 120) -> dict[str, Any]:
    if not math.isfinite(timeout_sec) or not 0 < timeout_sec <= 3600:
        raise ValueError("Timeout must be positive and no greater than 3600 seconds")
    output = Path(outdir).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    request = load_glyph_dataset(dataset, allowed_chars)
    result = subprocess.run([sys.executable, "-m", "zocr.consensus.learning", "--worker"],
                            input=_json_bytes(request), capture_output=True, timeout=timeout_sec)
    if result.returncode:
        raise RuntimeError("Isolated glyph learning failed: " + result.stderr.decode("utf-8", errors="replace")[-2000:])
    response = json.loads(result.stdout)
    output.mkdir(mode=0o700, parents=True)
    artifacts = {"learning_report.json": response["report"]}
    if response["memory"] is not None:
        artifacts["candidate_memory.json"] = response["memory"]
    for name, payload in artifacts.items():
        descriptor = os.open(output / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_json_bytes(payload))
    return response["report"]


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit("Use the consensus toy-learn subcommand to run a validated learning trial.")
    raw_request = sys.stdin.buffer.read(32 * 1024 * 1024 + 1)
    if len(raw_request) > 32 * 1024 * 1024:
        raise SystemExit("Oversized learning request")
    with io.StringIO() as diagnostics:
        from contextlib import redirect_stdout
        with redirect_stdout(diagnostics):
            response = _worker(json.loads(raw_request))
    sys.stdout.buffer.write(_json_bytes(response))
