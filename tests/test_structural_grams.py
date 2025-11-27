import json

from zocr.core.structural_grams import (
    classify_token,
    extract_cell_structural_gram,
    extract_structural_grams,
)


def test_classify_token_variants():
    assert classify_token("123.4") == "NUM"
    assert classify_token("A") == "UNIT"
    assert classify_token("MPa") == "UNIT"
    assert classify_token("rpm,") == "UNIT"
    assert classify_token("TAG-99") == "ID"
    assert classify_token("±") == "SYM"
    assert classify_token("(Φ)") == "SYM"
    assert classify_token("記号") == "TEXT"


def test_extract_cell_structural_gram():
    cell = {
        "text": "定格電流 100 A",
        "row_type": "body",
        "row_index": 12,
        "col_index": 3,
        "meta": {"filters": {"row_role": "body"}},
    }
    doc_meta = {"doc_id": "doc1", "table_id": "tbl-1", "cell_id": "cell-1"}
    gram = extract_cell_structural_gram(cell, doc_meta, max_ngram=5)

    assert gram["lexical_ngram"] == ["定格電流", "100", "A"]
    assert gram["type_ngram"][:3] == ["TEXT", "NUM", "UNIT"]
    assert gram["signatures"]["type_signature"] == "TEXT-NUM-UNIT"
    assert gram["signatures"]["type_signature_coalesced"] == "TEXT-NUM-UNIT"
    assert gram["layout"]["row_idx"] == 12
    assert gram["layout"]["col_idx"] == 3
    assert gram["layout"]["row_type"] == "body"
    assert gram["value_features"]["has_unit"]
    assert not gram["value_features"].get("has_symbol")
    assert gram["value_features"]["numeric_value"] == 100.0
    assert gram["value_features"]["unit"] == "A"
    assert gram["doc_meta"]["doc_id"] == "doc1"
    assert gram["doc_meta"]["table_id"] == "tbl-1"
    assert gram["doc_meta"]["cell_id"] == "cell-1"


def test_extract_structural_grams_reads_cells(tmp_path):
    records = [
        {
            "doc_id": "d1",
            "page": 1,
            "table_index": 0,
            "row": 0,
            "col": 1,
            "text": "PN-01 55 mm",
            "meta": {"filters": {"row_role": "header"}},
        },
        {
            "cell": {
                "doc_id": "d2",
                "page": 2,
                "row": 3,
                "col": 0,
                "text": "Φ 200〜250 mm",
                "row_type": "body",
            },
            "cell_id": "cell-2",
        },
    ]

    src = tmp_path / "cells.jsonl"
    out = tmp_path / "grams.jsonl"
    with open(src, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    count = extract_structural_grams(str(src), str(out), max_ngram=3)
    assert count == 2

    with open(out, "r", encoding="utf-8") as f:
        grams = [json.loads(line) for line in f if line.strip()]

    assert grams[0]["layout"]["row_type"] == "header"
    assert grams[0]["type_ngram"] == ["ID", "NUM", "UNIT"]
    assert grams[0]["value_features"]["has_unit"]
    assert grams[0]["doc_meta"]["doc_id"] == "d1"
    assert grams[0]["doc_meta"]["page"] == 1

    assert grams[1]["layout"]["row_type"] == "body"
    assert grams[1]["type_ngram"][0] == "SYM"
    assert grams[1]["value_features"].get("numeric_value") == 200.0
    assert grams[1]["value_features"]["has_symbol"]
    assert grams[1]["value_features"].get("range") == [200.0, 250.0]
    assert grams[1]["doc_meta"]["cell_id"] == "cell-2"
