import json
from pathlib import Path

import pytest

from zocr.core.indexer import build_index
from zocr.core.query_engine import hybrid_query

pytest.importorskip("numpy")


def _write_sample(tmp_path: Path) -> Path:
    records = [
        {
            "doc_id": "docA",
            "page": 1,
            "text": "Unit price PN-001",
            "search_unit": "Unit price",
            "zone": "table.body.row",
            "region_id": "tbl#1.row#1.col#unit_price",
            "struct": {
                "table_id": "tbl#1",
                "row": 1,
                "col": "unit_price",
                "header_norm": "unit_price",
                "row_key": {"part_no": "PN-001"},
            },
            "confidence": {"ocr": 0.95, "structure": 0.92},
            "meta": {"filters": {"amount": 10}},
        },
        {
            "doc_id": "docA",
            "page": 1,
            "text": "Amount total PN-002",
            "search_unit": "Amount total",
            "zone": "table.body.row",
            "region_id": "tbl#1.row#2.col#amount_total",
            "struct": {
                "table_id": "tbl#1",
                "row": 2,
                "col": "amount_total",
                "header_norm": "amount_total",
                "row_key": {"part_no": "PN-002"},
            },
            "confidence": {"ocr": 0.52, "structure": 0.48},
            "meta": {"filters": {"amount": 120}},
        },
        {
            "doc_id": "docA",
            "page": 1,
            "text": "Safety notes for assembly",
            "search_unit": "Safety notes",
            "zone": "section.body",
            "region_id": "sec#1",
            "struct": {"section": "safety"},
            "confidence": {"ocr": 0.99, "structure": 0.9},
            "meta": {},
        },
        {
            "doc_id": "docB",
            "page": 2,
            "text": "Unit price PN-777",
            "search_unit": "Unit price",
            "zone": "table.body.row",
            "region_id": "tbl#9.row#1.col#unit_price",
            "struct": {
                "table_id": "tbl#9",
                "row": 1,
                "col": "unit_price",
                "header_norm": "unit_price",
                "row_key": {"part_no": "PN-777"},
            },
            "confidence": {"ocr": 0.97, "structure": 0.95},
            "meta": {"filters": {"amount": 45}},
        },
    ]
    jsonl_path = tmp_path / "cells.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return jsonl_path


def test_hybrid_query_prioritizes_structural_boosts(tmp_path):
    jsonl_path = _write_sample(tmp_path)
    index_path = tmp_path / "ix.pkl"
    build_index(str(jsonl_path), str(index_path))

    results = hybrid_query(
        str(index_path),
        str(jsonl_path),
        q_text="unit price",
        zone_filter=r"table\\.body.*",
        boosts={"header_norm": ["unit_price"]},
        topk=3,
    )

    assert results
    top_score, top_doc = results[0]
    assert top_doc["region_id"] == "tbl#1.row#1.col#unit_price"
    retrieval = top_doc["meta"]["retrieval_scores"]
    assert retrieval["zone"] == 1.0
    assert retrieval["header"] >= 1.0
    assert top_score >= results[-1][0]


def test_hybrid_query_applies_struct_filters_and_penalty(tmp_path):
    jsonl_path = _write_sample(tmp_path)
    index_path = tmp_path / "ix.pkl"
    build_index(str(jsonl_path), str(index_path))

    filtered = hybrid_query(
        str(index_path),
        str(jsonl_path),
        q_text="amount total",
        zone_filter=r"table\\.body.*",
        struct_filters={"row": 2},
        topk=2,
    )

    assert len(filtered) == 1
    _, doc = filtered[0]
    assert doc["struct"]["row"] == 2
    assert doc["meta"]["retrieval_scores"]["penalty"] > 0.0


def test_hybrid_query_hard_filters_metadata(tmp_path):
    jsonl_path = _write_sample(tmp_path)
    index_path = tmp_path / "ix.pkl"
    build_index(str(jsonl_path), str(index_path))

    filtered = hybrid_query(
        str(index_path),
        str(jsonl_path),
        q_text="unit price",
        filters={"doc_id": "docB", "page": {2, 3}},
        topk=5,
    )

    assert filtered
    assert all(doc["doc_id"] == "docB" for _, doc in filtered)
    assert all(doc["page"] == 2 for _, doc in filtered)
