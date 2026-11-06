import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from zocr.consensus.learning import load_glyph_dataset, select_candidate, run_learning_trial


def dataset(tmp_path):
    rows = []
    font = ImageFont.load_default()
    for split, scale, group in (("train", 2, "training-document"), ("holdout", 3, "unseen-document-a"),
                                ("holdout", 4, "unseen-document-b")):
        for char in "2358":
            image = Image.new("L", (24, 24), 255)
            ImageDraw.Draw(image).text((4, 3), char, fill=0, font=font)
            image = image.resize((24 * scale, 24 * scale), Image.Resampling.LANCZOS)
            name = f"{split}-{scale}-{char}.png"
            image.save(tmp_path / name)
            rows.append({"id": name, "image": name, "text": char, "group_id": group, "split": split,
                         "approved": True, "reviewer": "fixture-owner", "reason": "Synthetic known label"})
    path = tmp_path / "dataset.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path, rows


def save_rows(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


@pytest.mark.parametrize("defect", ["unapproved", "group_overlap", "pixel_overlap", "missing", "unknown_split", "duplicate_id"])
def test_invalid_or_leaking_glyph_datasets_are_rejected(tmp_path, defect):
    path, rows = dataset(tmp_path)
    if defect == "unapproved":
        rows[0]["approved"] = False
    elif defect == "group_overlap":
        rows[-1]["group_id"] = rows[0]["group_id"]
    elif defect == "pixel_overlap":
        rows[-1]["image"] = rows[0]["image"]
        rows[-1]["text"] = rows[0]["text"]
    elif defect == "missing":
        rows[-1]["image"] = "missing.png"
    elif defect == "unknown_split":
        rows[-1]["split"] = "training"
    else:
        rows[-1]["id"] = rows[0]["id"]
    save_rows(path, rows)
    with pytest.raises(ValueError):
        load_glyph_dataset(path, "2358")


def test_candidate_requires_improvement_without_individual_regressions():
    baseline = [{"correct": True}, {"correct": False}, {"correct": False}]
    assert select_candidate(baseline, [{"correct": True}] * 3)["eligible"]
    assert not select_candidate(baseline, baseline)["eligible"]
    assert not select_candidate(baseline, [{"correct": False}, {"correct": True}, {"correct": True}])["eligible"]


def test_real_glyph_learning_trial_is_isolated_and_auditable(tmp_path):
    from zocr.consensus import toy_runtime

    path, _ = dataset(tmp_path)
    before = toy_runtime._toy_memory_payload()
    output = tmp_path / "trial"
    report = run_learning_trial(path, output, allowed_chars="2358", timeout_sec=60)
    assert report["scope"] == "isolated_glyph_matching"
    assert report["baseline"]["samples"] == report["candidate"]["samples"] == 8
    assert report["dataset_sha256"]
    assert report["baseline_model_sha256"]
    assert report["selection"]["regressions"] >= 0
    assert toy_runtime._toy_memory_payload() == before
    saved = json.loads((output / "learning_report.json").read_text(encoding="utf-8"))
    assert saved == report
    model = output / "candidate_memory.json"
    assert model.exists() == report["selection"]["eligible"]
    if model.exists():
        memory = json.loads(model.read_text(encoding="utf-8"))
        assert memory["learning_trial"]["dataset_sha256"] == report["dataset_sha256"]
    with pytest.raises(FileExistsError):
        run_learning_trial(path, output, allowed_chars="2358")
    print(json.dumps({key: report[key] for key in ("baseline", "candidate", "selection")}, sort_keys=True))


def test_learning_cli_runs_without_precreating_the_trial_directory(tmp_path):
    import subprocess
    import sys

    path, _ = dataset(tmp_path)
    result = subprocess.run([sys.executable, "-m", "zocr.consensus.zocr_consensus", "toy-learn",
                             "--dataset", str(path), "--allowed-chars", "2358", "--out", str(tmp_path / "cli-trial")],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1])["baseline"]["samples"] == 8


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), 3601])
def test_invalid_trial_timeout_does_not_create_outputs(tmp_path, timeout):
    with pytest.raises(ValueError):
        run_learning_trial(tmp_path / "missing.jsonl", tmp_path / "trial", allowed_chars="2358", timeout_sec=timeout)
    assert not (tmp_path / "trial").exists()


def test_dataset_paths_cannot_escape_through_symlinks(tmp_path):
    path, rows = dataset(tmp_path)
    outside = tmp_path.parent / (tmp_path.name + "-outside.png")
    Image.new("L", (10, 10), 255).save(outside)
    link = tmp_path / "linked.png"
    link.symlink_to(outside)
    rows[0]["image"] = link.name
    save_rows(path, rows)
    with pytest.raises(ValueError, match="outside"):
        load_glyph_dataset(path, "2358")


def test_template_cache_is_pixel_keyed_and_bounded():
    import numpy as np
    from zocr.consensus import toy_runtime as runtime

    cached = runtime._cached_glyph_norms
    cached.cache_clear()
    glyph = runtime._GLYPH_ATLAS["5"][0].copy()
    key = (glyph.tobytes(), glyph.size, (24, 32))
    first = cached(*key)
    assert cached(*key) is first
    assert cached.cache_info().hits == 1
    assert cached.cache_info().maxsize == 128
    assert all(not array.flags.writeable for array in first)
    assert all(np.array_equal(a, b) for a, b in zip(first, cached.__wrapped__(*key)))
    glyph.paste(0, (0, 0, *glyph.size))
    changed = cached(glyph.tobytes(), glyph.size, (24, 32))
    assert not np.array_equal(first[4], changed[4])
    size = cached.cache_info().currsize
    runtime._match_glyph(glyph.resize((80, 80)), runtime._GLYPH_ATLAS, allowed_chars="5")
    assert cached.cache_info().currsize == size


def test_template_cache_preserves_results_and_reports_latency():
    import time
    from statistics import median
    from zocr.consensus import toy_runtime as runtime

    glyph = runtime._GLYPH_ATLAS["5"][0]
    elapsed = {"cold": [], "warm": []}
    results = []
    for mode in elapsed:
        for _ in range(20):
            if mode == "cold":
                runtime._cached_glyph_norms.cache_clear()
            started = time.perf_counter()
            results.append(runtime._match_glyph(glyph, runtime._GLYPH_ATLAS, allowed_chars="2358"))
            elapsed[mode].append((time.perf_counter() - started) * 1000)
    assert all(result == results[0] for result in results)
    print(json.dumps({"template_cache_median_ms": {mode: median(values) for mode, values in elapsed.items()}}))


def test_learning_cli_registers_without_changing_default_ocr():
    import argparse
    from zocr.consensus.cli import _patch_cli_for_export_and_search

    parser = _patch_cli_for_export_and_search(argparse.ArgumentParser())
    args = parser.parse_args(["toy-learn", "--dataset", "glyphs.jsonl", "--allowed-chars", "0123456789", "--out", "new-trial"])
    assert args.cmd == "toy-learn"
    assert Path(args.out).name == "new-trial"
