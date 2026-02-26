from __future__ import annotations

import json


def test_toy_bench_jsonl_keeps_blank_expected(tmp_path) -> None:
    from zocr.consensus.cli import _iter_toy_bench_samples

    (tmp_path / "a.png").write_bytes(b"")
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(json.dumps({"image": "a.png", "text": ""}) + "\n", encoding="utf-8")

    samples = _iter_toy_bench_samples(str(dataset_path))
    assert len(samples) == 1
    assert samples[0]["expected"] == ""

