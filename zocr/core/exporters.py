"""Export helpers for SQL and RAG bundles."""
from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from .domains import DOMAIN_ALIAS, DOMAIN_SUGGESTED_QUERIES

__all__ = ["sql_export", "export_rag_bundle", "_build_trace", "_fact_tag"]


def sql_export(jsonl: str, outdir: str, prefix: str = "invoice") -> Dict[str, str]:
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, f"{prefix}_cells.csv")
    schema_path = os.path.join(outdir, f"{prefix}_schema.sql")
    cols = [
        "doc_id",
        "page",
        "table_index",
        "row",
        "col",
        "text",
        "search_unit",
        "synthesis_window",
        "amount",
        "date",
        "company",
        "address",
        "tax_id",
        "postal_code",
        "phone",
        "tax_rate",
        "qty",
        "unit",
        "subtotal",
        "tax_amount",
        "corporate_id",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "confidence",
        "low_conf",
        "phash64",
        "lambda_shape",
        "trace",
    ]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as fw:
        wr = csv.writer(fw)
        wr.writerow(cols)
        with open(jsonl, "r", encoding="utf-8") as fr:
            for line in fr:
                ob = json.loads(line)
                meta = ob.get("meta") or {}
                filt = meta.get("filters", {})
                x1, y1, x2, y2 = ob.get("bbox", [0, 0, 0, 0])
                wr.writerow(
                    [
                        ob.get("doc_id"),
                        ob.get("page"),
                        ob.get("table_index"),
                        ob.get("row"),
                        ob.get("col"),
                        ob.get("text"),
                        ob.get("search_unit"),
                        ob.get("synthesis_window"),
                        filt.get("amount"),
                        filt.get("date"),
                        filt.get("company"),
                        filt.get("address"),
                        filt.get("tax_id"),
                        filt.get("postal_code"),
                        filt.get("phone"),
                        filt.get("tax_rate"),
                        filt.get("qty"),
                        filt.get("unit"),
                        filt.get("subtotal"),
                        filt.get("tax_amount"),
                        filt.get("corporate_id"),
                        x1,
                        y1,
                        x2,
                        y2,
                        meta.get("confidence"),
                        meta.get("low_conf"),
                        meta.get("phash64"),
                        meta.get("lambda_shape"),
                        filt.get("trace"),
                    ]
                )
    schema = f"""
CREATE TABLE IF NOT EXISTS {prefix}_cells (
  doc_id TEXT, page INT, table_index INT, row INT, col INT,
  text TEXT, search_unit TEXT, synthesis_window TEXT,
  amount BIGINT, date TEXT, company TEXT, address TEXT, tax_id TEXT,
  postal_code TEXT, phone TEXT, tax_rate REAL, qty BIGINT, unit TEXT, subtotal BIGINT, tax_amount BIGINT, corporate_id TEXT,
  bbox_x1 INT, bbox_y1 INT, bbox_x2 INT, bbox_y2 INT,
  confidence REAL, low_conf BOOLEAN, phash64 BIGINT, lambda_shape REAL, trace TEXT
);
-- COPY {prefix}_cells FROM '{csv_path}' WITH CSV HEADER;
"""
    with open(schema_path, "w", encoding="utf-8") as fw:
        fw.write(schema)
    return {"csv": csv_path, "schema": schema_path}


def _build_trace(
    doc_id: Optional[str],
    page: int,
    table_idx: Optional[int],
    row_idx: Optional[int],
    col_idx: Optional[int],
    bbox: List[Any],
) -> Tuple[Dict[str, Any], str]:
    trace_dict = {
        "doc_id": doc_id,
        "page": int(page) if page is not None else None,
        "table_index": int(table_idx) if table_idx is not None else None,
        "row": int(row_idx) if row_idx is not None else None,
        "col": int(col_idx) if col_idx is not None else None,
        "bbox": bbox,
    }
    label_parts = [
        f"doc={doc_id if doc_id is not None else 'NA'}",
        f"page={page if page is not None else 'NA'}",
        f"table={table_idx if table_idx is not None else 'NA'}",
        f"row={row_idx if row_idx is not None else 'NA'}",
        f"col={col_idx if col_idx is not None else 'NA'}",
    ]
    trace_label = ";".join(label_parts)
    return trace_dict, trace_label


def _fact_tag(text: str, trace_label: str, lang: Optional[str]) -> str:
    payload = {"trace": trace_label, "lang": lang or ""}
    attrs = " ".join(
        f"{k}={json.dumps(v)}" for k, v in payload.items() if v is not None and v != ""
    )
    body = (text or "").replace("<", "&lt;").replace(">", "&gt;")
    return f"<fact {attrs}>{body}</fact>"


def _bedrock_embedding_hint(model: str) -> Dict[str, Any]:
    cli_body = '{"inputText": "hello world"}'
    return {
        "region_env": "AWS_REGION",
        "payload_key": "inputText",
        "content_type": "application/json",
        "accept": "application/json",
        "cli_example": {
            "command": [
                "aws",
                "bedrock-runtime",
                "invoke-model",
                "--region",
                "${AWS_REGION}",
                "--model-id",
                model,
                "--body",
                cli_body,
            ],
            "body": cli_body,
        },
        "python_example": """
import json, os
import boto3

client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))
payload = {"inputText": "hello world"}
res = client.invoke_model(modelId="%(model)s", body=json.dumps(payload))
vector = json.loads(res["body"].read())["embedding"]
"""
        % {"model": model},
    }


def export_rag_bundle(
    jsonl: str,
    outdir: str,
    domain: Optional[str] = None,
    summary: Optional[Dict[str, Any]] = None,
    limit_per_section: int = 40,
    embedding_provider: str = "aws_bedrock",
    embedding_model: str = "amazon.titan-embed-text-v2",
) -> Dict[str, Any]:
    if not os.path.exists(jsonl):
        raise FileNotFoundError(jsonl)
    os.makedirs(outdir, exist_ok=True)
    cells_path = os.path.join(outdir, "cells.jsonl")
    sections_path = os.path.join(outdir, "sections.jsonl")
    tables_path = os.path.join(outdir, "tables.json")
    manifest_path = os.path.join(outdir, "manifest.json")
    markdown_path = os.path.join(outdir, "bundle.md")

    resolved = DOMAIN_ALIAS.get(domain, domain) if domain else None
    if not resolved:
        resolved = "invoice_jp_v2"
    suggested = DOMAIN_SUGGESTED_QUERIES.get(resolved, DOMAIN_SUGGESTED_QUERIES["default"])

    doc_ids: set = set()
    languages: set = set()
    page_sections: Dict[int, Dict[str, Any]] = {}
    tables: Dict[str, Dict[str, Any]] = {}
    cells_written = 0

    with open(jsonl, "r", encoding="utf-8") as fr, open(
        cells_path, "w", encoding="utf-8"
    ) as fw:
        for line in fr:
            ob = json.loads(line)
            fw.write(json.dumps(ob, ensure_ascii=False) + "\n")
            doc_ids.add(ob.get("doc_id"))
            lang = (ob.get("meta") or {}).get("lang")
            if lang:
                languages.add(lang)
            page_sections.setdefault(ob.get("page"), {"cells": []})["cells"].append(ob)
            table_key = f"{ob.get('doc_id')}:{ob.get('table_index')}"
            tables.setdefault(table_key, {"rows": []})["rows"].append(ob)
            cells_written += 1

    sections: List[Dict[str, Any]] = []
    for page, info in sorted(page_sections.items()):
        entries = info["cells"][:limit_per_section]
        text_blob = "\n".join((cell.get("text") or "") for cell in entries)
        sections.append(
            {
                "page": page,
                "cell_count": len(entries),
                "text": text_blob,
            }
        )
    with open(sections_path, "w", encoding="utf-8") as fw:
        for sec in sections:
            fw.write(json.dumps(sec, ensure_ascii=False) + "\n")

    table_manifest = {k: v for k, v in tables.items()}
    with open(tables_path, "w", encoding="utf-8") as fw:
        json.dump(table_manifest, fw, ensure_ascii=False, indent=2)

    trace_schema = {
        "label": "doc=<str>;page=<int>;table=<int>;row=<int>;col=<int>",
        "fields": {
            "doc_id": "Document identifier, if provided",
            "page": "1-indexed page number where the cell originated",
            "table_index": "0-indexed table identifier or null for non-tabular cells",
            "row": "0-indexed row index within the table",
            "col": "0-indexed column index within the table",
            "bbox": "Bounding box [x1, y1, x2, y2] in pixel coordinates",
        },
    }
    sample_trace = _build_trace(None, 1, 0, 0, 0, [0, 0, 10, 10])[1]
    fact_tag_example = _fact_tag("amount=12,000", sample_trace, lang=list(languages)[0] if languages else None)
    embedding = {
        "provider": embedding_provider,
        "model": embedding_model,
        "note": "Preferred RAG embedding target; swap model for region-specific Bedrock deployments as needed.",
    }
    if embedding_provider == "aws_bedrock":
        embedding["hint"] = _bedrock_embedding_hint(embedding_model)

    manifest = {
        "cells": cells_path,
        "sections": sections_path,
        "tables": tables_path,
        "suggested_queries": suggested,
        "doc_ids": list(doc_ids),
        "languages": sorted(languages),
        "summary": summary,
        "cells_written": cells_written,
        "page_sections": len(sections),
        "table_sections": len(tables),
        "trace_schema": trace_schema,
        "fact_tag_example": fact_tag_example,
        "embedding": embedding,
    }
    with open(manifest_path, "w", encoding="utf-8") as fw:
        json.dump(manifest, fw, ensure_ascii=False, indent=2)

    with open(markdown_path, "w", encoding="utf-8") as fw:
        fw.write(f"# ZOCR RAG Bundle ({resolved})\n\n")
        fw.write("## Suggested Queries\n")
        for q in suggested:
            fw.write(f"- {q}\n")
        fw.write("\n## Sections\n")
        for sec in sections:
            fw.write(f"### Page {sec['page']} ({sec['cell_count']} cells)\n")
            fw.write(sec["text"] + "\n\n")

    return {
        "cells": cells_path,
        "sections": sections_path,
        "tables": tables_path,
        "tables_json": tables_path,
        "manifest": manifest_path,
        "bundle_dir": outdir,
        "markdown": markdown_path,
        "cell_count": cells_written,
        "table_sections": len(tables),
        "page_sections": len(sections),
        "doc_ids": list(doc_ids),
        "languages": sorted(languages),
        "suggested_queries": suggested,
        "trace_schema": trace_schema,
        "fact_tag_example": fact_tag_example,
        "embedding": embedding,
    }
