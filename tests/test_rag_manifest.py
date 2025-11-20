import json

from zocr.core.exporters import export_rag_bundle


def test_export_rag_bundle_embeds_bedrock_plan(tmp_path):
    cells_path = tmp_path / "cells.jsonl"
    cells_path.write_text(
        json.dumps(
            {
                "doc_id": "DOC-1",
                "page": 1,
                "table_index": 0,
                "row": 0,
                "col": 0,
                "text": "amount=12000",
                "meta": {"lang": "ja"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = export_rag_bundle(str(cells_path), str(tmp_path / "rag"), domain="invoice")

    with open(result["manifest"], "r", encoding="utf-8") as fr:
        manifest = json.load(fr)

    assert result["embedding"]["provider"] == "aws_bedrock"
    assert result["embedding"]["model"] == "amazon.titan-embed-text-v2"
    assert manifest["embedding"] == result["embedding"]

    assert result["cell_count"] == 1
    assert result["page_sections"] == 1
    assert result["table_sections"] == 1
    assert manifest["trace_schema"]["label"].startswith("doc=")
    assert "fact_tag_example" in manifest and manifest["fact_tag_example"]

