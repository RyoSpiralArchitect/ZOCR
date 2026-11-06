from __future__ import annotations

import base64
import hashlib
import io
import json
import math
from pathlib import Path
import os
import sys
import tempfile
import time

from PIL import Image

from .learning import _json_bytes, _read_bounded
from ..service.executor import execute_command


def edit_distance(expected, actual):
    if len(expected) < len(actual):
        expected, actual = actual, expected
    previous = list(range(len(actual) + 1))
    for i, left in enumerate(expected, 1):
        current = [i]
        for j, right in enumerate(actual, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def text_metrics(pairs):
    if not pairs:
        raise ValueError("Evaluation requires at least one sample")
    chars = words = char_edits = word_edits = exact = 0
    for expected, predicted in pairs:
        chars += len(expected)
        words += len(expected.split())
        char_edits += edit_distance(expected, predicted)
        word_edits += edit_distance(expected.split(), predicted.split())
        exact += expected == predicted
    return {"samples": len(pairs), "exact_matches": exact, "exact_match_rate": exact / len(pairs),
            "reference_characters": chars, "character_edits": char_edits,
            "cer": char_edits / chars if chars else None,
            "reference_words": words, "word_edits": word_edits, "wer": word_edits / words if words else None,
            "word_tokenization": "Unicode whitespace; not linguistic Japanese word segmentation"}


def compare_cell_evaluations(baseline, candidate):
    old, new = baseline["predictions"], candidate["predictions"]
    if [row["id"] for row in old] != [row["id"] for row in new]:
        raise ValueError("Evaluation sample identities do not match")
    regressions = []
    for a, b in zip(old, new):
        if a["expected"] != b["expected"]:
            raise ValueError("Evaluation reference labels changed")
        raw = a["expected"]
        if (edit_distance(raw, b["predicted"]) > edit_distance(raw, a["predicted"])
                or edit_distance(raw.split(), b["predicted"].split()) > edit_distance(raw.split(), a["predicted"].split())):
            regressions.append(a["id"])
    return {"scope": "independent_full_cell", "regressions": regressions,
            "recommended_action": "reject" if regressions else "manual_review_only", "activation_supported": False}


def load_cell_dataset(path, memory=None):
    path = Path(path).resolve()
    raw = _read_bounded(path, 2 * 1024 * 1024)
    if memory is not None and not isinstance(memory, dict):
        raise ValueError("Expected a full frozen memory object")
    lineage = (memory or {}).get("learning_trial", {})
    if memory is not None and (not lineage.get("document_lineage_complete") or not lineage.get("training_document_sha256")):
        raise ValueError("Candidate memory lacks document-level training lineage")
    excluded = set(lineage.get("training_document_sha256", [])) | set(lineage.get("selection_document_sha256", []))
    rows, ids, pixels, docs = [], set(), set(), {}
    for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict) or row.get("split") != "test" or row.get("approved") is not True:
                raise ValueError("Only approved independent test records are accepted")
            for key in ("id", "image", "document", "group_id", "reviewer", "reason"):
                if not isinstance(row.get(key), str) or not 1 <= len(row[key].strip()) <= 512:
                    raise ValueError(f"Invalid {key}")
            if not isinstance(row.get("text"), str) or len(row["text"]) > 1000:
                raise ValueError("Expected bounded raw text, including empty strings")
            if row.get("source_kind") not in {"fixture", "real_document"}:
                raise ValueError("Declare fixture or real_document provenance")
            resolved = [(path.parent / row[key]).resolve() for key in ("image", "document")]
            if any(not candidate.is_relative_to(path.parent) for candidate in resolved):
                raise ValueError("Evaluation files must remain inside the dataset directory")
            image_path, document = resolved
            if document not in docs:
                docs[document] = hashlib.sha256(_read_bounded(document, 64 * 1024 * 1024)).hexdigest()
            document_hash = docs[document]
            excluded_groups = set(lineage.get("training_group_ids", [])) | set(lineage.get("selection_group_ids", []))
            if document_hash in excluded or row.get("group_id", document_hash) in excluded_groups:
                raise ValueError("Test document was used for training or candidate selection")
            data = _read_bounded(image_path, 4 * 1024 * 1024)
            with Image.open(io.BytesIO(data)) as image:
                if image.width * image.height > 1_000_000:
                    raise ValueError("Cell crop exceeds one million pixels")
                image = image.convert("L")
                pixel_hash = hashlib.sha256(_json_bytes(image.size) + image.tobytes()).hexdigest()
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
            if row["id"] in ids or pixel_hash in pixels:
                raise ValueError("Repeated evaluation IDs or pixels")
            ids.add(row["id"])
            pixels.add(pixel_hash)
            rows.append({"id": row["id"], "expected": row["text"], "document_sha256": document_hash,
                         "source_kind": row["source_kind"], "pixels_sha256": pixel_hash,
                         "annotation_sha256": hashlib.sha256(_json_bytes(row)).hexdigest(),
                         "png": base64.b64encode(buffer.getvalue()).decode("ascii")})
            if len(rows) > 100:
                raise ValueError("At most 100 independent cells per evaluation")
        except (ValueError, OSError, TypeError, Image.DecompressionBombError) as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
    if not rows:
        raise ValueError("Empty independent test set")
    return {"dataset_sha256": hashlib.sha256(raw).hexdigest(), "samples": rows}


def evaluate_cells(dataset, outdir, *, memory=None, allowed_chars=None, timeout_sec=120):
    if not math.isfinite(timeout_sec) or not 0 < timeout_sec <= 3600:
        raise ValueError("Invalid evaluation deadline")
    if allowed_chars is not None and (not isinstance(allowed_chars, str) or not 1 <= len(allowed_chars) <= 256):
        raise ValueError("Invalid explicit alphabet")
    output = Path(outdir).resolve()
    if output.exists():
        raise FileExistsError("Evaluation output must be new")
    snapshot = load_cell_dataset(dataset, memory)
    predictions = []
    deadline = time.monotonic() + timeout_sec
    for sample in snapshot["samples"]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Cell evaluation deadline exceeded")
        prediction = execute_command([sys.executable, "-m", "zocr.consensus.cell_evaluation", "--worker", str(os.getpid())],
                                     {"png": sample["png"], "memory": memory, "alphabet": allowed_chars}, remaining)
        predictions.append({**{key: value for key, value in sample.items() if key != "png"}, **prediction})
    report = {"schema": "zocr.cell_evaluation.v1", "scope": "full_cell_ocr", "split": "independent_test",
              "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "dataset_sha256": snapshot["dataset_sha256"], "model_sha256": hashlib.sha256(_json_bytes(memory)).hexdigest(),
              "model": "explicit_memory" if memory else "builtin_atlas", "allowed_chars": allowed_chars,
              "metrics": text_metrics([(row["expected"], row["predicted"]) for row in predictions]),
              "predictions": predictions, "training_or_selection": False,
              "real_document_samples": sum(row["source_kind"] == "real_document" for row in predictions),
              "limitations": ["Source kind and gold labels are asserted by the dataset owner.",
                              "Repeated inspection of final-test results can bias later model selection.",
                              "Each cell runs in a fresh interpreter; no cross-cell adaptive state or production activation."]}
    output.mkdir(mode=0o700, parents=True)
    with (output / "cell_evaluation.json").open("xb") as stream:
        stream.write(_json_bytes(report))
    return report


def _worker(request):
    from . import toy_runtime as runtime

    with tempfile.TemporaryDirectory(prefix="zocr-frozen-cell-") as directory:
        if request["memory"] is not None:
            path = Path(directory) / "memory.json"
            path.write_bytes(_json_bytes(request["memory"]))
            for state in (runtime._GLYPH_ATLAS, runtime._GLYPH_FEATS, runtime._NGRAM_COUNTS, runtime._NGRAM_TOTALS):
                state.clear()
            if not runtime.load_toy_memory(str(path)).get("loaded"):
                raise ValueError("Frozen model could not be loaded")
            effective = runtime._toy_memory_payload()
            if _json_bytes(effective) != _json_bytes({key: request["memory"].get(key) for key in effective}):
                raise ValueError("Frozen model changed while loading; evaluation refused")
        effective_hash = hashlib.sha256(_json_bytes(runtime._toy_memory_payload())).hexdigest()
        with Image.open(io.BytesIO(base64.b64decode(request["png"], validate=True))) as image:
            started = time.perf_counter()
            text, confidence = runtime.toy_ocr_text_from_cell(image, allowed_chars=request["alphabet"])
            elapsed = time.perf_counter() - started
    return {"predicted": text, "confidence": confidence, "recognition_seconds": elapsed,
            "effective_model_sha256": effective_hash,
            "configuration_sha256": hashlib.sha256(json.dumps(runtime.toy_runtime_config(), sort_keys=True, default=str).encode()).hexdigest(),
            "runtime_sha256": hashlib.sha256(Path(runtime.__file__).read_bytes()).hexdigest()}


if __name__ == "__main__":
    from contextlib import redirect_stdout
    from ..service.executor import watch_parent
    if len(sys.argv) != 3 or sys.argv[1] != "--worker":
        raise SystemExit("Use the consensus cell-eval subcommand")
    watch_parent(int(sys.argv[2]))
    raw = sys.stdin.buffer.read(32 * 1024 * 1024 + 1)
    if len(raw) > 32 * 1024 * 1024:
        raise SystemExit("Oversized evaluation request")
    with redirect_stdout(io.StringIO()):
        response = _worker(json.loads(raw))
    sys.stdout.buffer.write(_json_bytes(response))
