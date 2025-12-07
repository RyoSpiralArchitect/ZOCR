import base64
import json
from pathlib import Path

import pytest

from zocr.core.indexer import build_index
from zocr.core.query_engine import hybrid_query

pytest.importorskip("numpy")


_FAKE_IMG_B64 = base64.b64encode(b"fakeimg").decode()


def _write_visual_sample(tmp_path: Path) -> Path:
    records = [
        {
            "doc_id": "visA",
            "page": 1,
            "text": "System overview diagram",
            "search_unit": "System overview diagram",
            "meta": {
                "object_type": "figure",
                "caption": "Hydraulic flow diagram",
                "thumbnail_b64": _FAKE_IMG_B64,
                "embeddings": {"hybrid": [0.9, 0.05, 0.05], "text": [0.8, 0.1, 0.1]},
            },
        },
        {
            "doc_id": "visA",
            "page": 2,
            "text": "Bill of materials table",
            "search_unit": "Bill of materials table",
            "zone": "table.body.row",
            "struct": {"table_id": "tbl#parts", "row": 1, "col": "part"},
            "meta": {
                "caption": "Parts table",
                "table_html": "<table><tr><td>bolt</td><td>4</td></tr></table>",
                "table_csv": "part,qty\nbolt,4",
                "embeddings": {"hybrid": [0.1, 0.8, 0.1], "text": [0.1, 0.7, 0.2]},
            },
        },
        {
            "doc_id": "visA",
            "page": 3,
            "text": "Safety notice paragraph",
            "search_unit": "Safety notice paragraph",
            "meta": {"embeddings": {"hybrid": [0.2, 0.2, 0.6], "text": [0.2, 0.2, 0.6]}},
        },
    ]
    jsonl_path = tmp_path / "visual.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return jsonl_path


def test_diagram_query_prefers_figure_and_payload(tmp_path):
    jsonl_path = _write_visual_sample(tmp_path)
    ix_path = tmp_path / "ix.pkl"
    build_index(str(jsonl_path), str(ix_path))

    results = hybrid_query(
        str(ix_path),
        str(jsonl_path),
        q_text="show me the diagram layout",
        hybrid_embedding=[0.9, 0.05, 0.05],
        topk=3,
    )

    assert results
    _, doc = results[0]
    meta = doc["meta"]
    assert meta["object_type"] == "figure"
    assert meta["render"]["thumbnail_b64"] == _FAKE_IMG_B64
    assert meta["render"].get("caption")


def test_table_query_returns_table_payload(tmp_path):
    jsonl_path = _write_visual_sample(tmp_path)
    ix_path = tmp_path / "ix.pkl"
    build_index(str(jsonl_path), str(ix_path))

    results = hybrid_query(
        str(ix_path),
        str(jsonl_path),
        q_text="need the parts table",
        hybrid_embedding=[0.1, 0.8, 0.1],
        topk=2,
    )

    assert results
    _, doc = results[0]
    meta = doc["meta"]
    assert meta["object_type"] == "table"
    assert meta["render"].get("html")
    assert meta["render"].get("csv")
    assert meta["render"].get("caption")
    assert meta["object_type"] != "text"
