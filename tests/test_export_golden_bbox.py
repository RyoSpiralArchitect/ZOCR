from __future__ import annotations

import json

from PIL import Image


def test_export_jsonl_bbox_golden(tmp_path) -> None:
    from zocr.consensus import toy_runtime

    x1, y1, x2, y2 = 10, 10, 210, 210
    col_bounds = [0, 100, 200]
    row_bands_rel = [(0, 40), (40, 80), (80, 120), (120, 160), (160, 200)]
    expected = {}
    for r, (rt, rb) in enumerate(row_bands_rel):
        for c in range(len(col_bounds) - 1):
            expected[(r, c)] = [x1 + col_bounds[c], y1 + rt, x1 + col_bounds[c + 1], y1 + rb]

    img_path = tmp_path / "page.png"
    Image.new("RGB", (220, 220), (255, 255, 255)).save(img_path)

    doc_path = tmp_path / "doc.zocr.json"
    doc_path.write_text(
        json.dumps(
            {
                "doc_id": "doc",
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
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    out_path = tmp_path / "out.jsonl"
    n = toy_runtime.export_jsonl_with_ocr(
        str(doc_path),
        str(img_path),
        str(out_path),
        ocr_engine="toy",
        contextual=False,
    )
    assert n == 10

    records = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(records) == 10
    seen = {(int(r["row"]), int(r["col"])): r["bbox"] for r in records}
    assert set(seen) == set(expected)
    for key, bbox in expected.items():
        assert seen[key] == bbox

