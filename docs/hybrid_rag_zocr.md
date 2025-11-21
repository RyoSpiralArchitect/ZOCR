<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (C) 2024 ZOCR contributors -->

# Hybrid RAG with Z-OCR semantics

This note outlines how to implement a structure-aware Retrieval-Augmented Generation (RAG) pipeline that uses Z-OCR tags so the retriever and reader reason over document layout as well as text.

## Goals
- Preserve document structure (tables, headers, zones, captions, keys) alongside OCR text.
- Enable retrieval that can filter and boost by semantic roles (e.g., table column, figure caption) instead of only fuzzy text.
- Support agentic follow-up tools to fetch precise cells/rows/figures during answer generation.

## Tag envelope (per chunk)
Each chunk produced by Z-OCR is wrapped with a minimal envelope (extend as needed):

```json
{
  "doc_id": "AI_テスト用_部品表1.pdf",
  "page": 4,
  "bbox": [x0, y0, x1, y1],
  "zone": "table.body.row",
  "region_id": "tbl#2.row#7.col#part_no",
  "struct": {
    "table_id": "tbl#2",
    "row": 7,
    "col": "part_no",
    "header_norm": "part_number",
    "row_key": {"assembly_id": "A-0421"}
  },
  "datatypes": ["alnum", "sku"],
  "units": null,
  "lang": "ja",
  "viz": {
    "phash": "a19c…",
    "lineage": ["scan", "dewarp:v0.3", "table_detect:v2.1"]
  },
  "confidence": { "ocr": 0.93, "structure": 0.88 },
  "text": "PN-88341",
  "embeddings": { "text": "…", "layout": "…" }
}
```

Key fields:
- **zone** and **struct** identify whether the chunk is a header cell, caption, list item, etc.
- **header_norm** aligns column names to a canonical schema for tabular queries.
- **phash** enables visual diff/versioning.
- **embeddings** may include separate text and layout vectors (or a concatenated multimodal embedding).

## Indexing strategy
- **Dense vectors:** sentence-level text embeddings plus optional layout embeddings (relative position, neighbors, font cues).
- **Sparse:** BM25 over normalized text and header/field names.
- **Symbolic filters:** exact matches over structural fields (e.g., `table_id = tbl#2 AND col = part_no`).
- **Graph edges:** connect cells by `same_row`, `header_of`, `caption_of`, and `next_page_continuation` to support neighbor expansion.

## Retrieval flow (agentic)
1. **Query parsing:** detect whether the ask targets a field, section, or figure; map synonyms to `header_norm` via a canonical dictionary.
2. **Stage-1 recall:** run BM25 over text + headers and dense ANN over text+layout embeddings.
3. **Structure-aware re-rank:** boost chunks whose tags match the inferred zone (e.g., `table.*` for tabular facts), penalize low OCR/structure confidence, and prefer `caption_of` when the query mentions figures/diagrams.
4. **Neighborhood graft:** if a top chunk is a table cell, also pull its row and header neighbors.
5. **Answer planning:** let the LLM inspect tags and call tools such as `fetch_cell` to retrieve the exact cell/row/caption before generating.
6. **Grounded generation:** compose the answer with cited cells/sections, including page/region/bbox, and normalize units when available.

A simple combined score for chunk `c`:

\[
S(c) = \alpha \cdot \text{DenseSim} + \beta \cdot \text{BM25} + \gamma \cdot \mathbf{1}[\text{zone match}] + \delta \cdot \text{HeaderMatch} + \eta \cdot \text{NeighborBoost} - \lambda \cdot (1 - \text{conf\_ocr})(1 - \text{conf\_struct})
\]

Typical starting weights: `alpha=0.4`, `beta=0.25`, `gamma=0.2`, `delta=0.1`, `eta=0.1`, `lambda=0.2`.

## Minimal components
- **Z-OCR pipeline:** emits text lines, reading order, tables, headers, figure/caption links, field normalization, per-chunk bounding boxes, and confidences.
- **Canonical dictionary:** e.g., `{ "vendor": ["supplier", "供給者"], "part_number": ["pn", "品番"] }` for header normalization.
- **Hybrid indexer:** vector store + BM25 with tags retained in a filterable store.
- **Retriever:** two-stage recall + re-ranker using tag features plus neighbor expansion.
- **Refine tools:** `get_cell(row_key, header_norm)`, `get_caption(figure_id)`, `crop_image(bbox)` to re-pull precise evidence.
- **Evaluator:** cell-level exact match, section-level overlap, citation correctness, and layout-sensitive F1 (avoid wrong-row credit).

## Tiny pseudo-API
```python
cands = retrieve_hybrid(
  q="単価と合計金額を出して",
  filters={"zone": "table.*"},
  boosts={"header_norm": ["unit_price", "amount_total"]},
  k=40)

cell = fetch_cell(
  doc_id=cands[0].doc_id,
  table_id=cands[0].struct["table_id"],
  row_key={"assembly_id": "A-0421"},
  header_norm="unit_price")
```

### Reference helper in the codebase
The repository now ships a light-weight re-ranker at `zocr.core.query_engine.hybrid_query`:

```python
from zocr.core import build_index, hybrid_query

ix = build_index("rag/cells.jsonl", "rag/index.pkl")
hits = hybrid_query(
    "rag/index.pkl",
    "rag/cells.jsonl",
    q_text="unit price and total",
    zone_filter=r"table\\.body.*",
    boosts={"header_norm": ["unit_price", "amount_total"]},
    filters={"doc_id": "AI_テスト用_部品表1.pdf", "page": {3, 4}},
)
```

It applies the same scoring recipe described above, boosting zone/header matches, honoring structural filters, respecting top-level/meta filters (e.g., ``doc_id``/``page``), and penalizing low OCR/structure confidence when those signals are present in the Z-OCR tag envelope.

## Evaluation tips
- **Cell QA:** ask for values present only in specific rows; measure exact match and wrong-row rate.
- **Section QA:** e.g., “summarize safety notes” and check section alignment.
- **Robustness:** add image skew/noise to confirm tag-guided retrieval still hits correct rows.
- **Latency targets:** aim for Stage-1 recall < 80 ms, re-rank < 40 ms, neighbor graft < 20 ms on local hardware.

## Prompting pattern for the reader LLM
- You will receive structured chunks with tags.
- If the question targets a table field, call `fetch_cell` to retrieve the exact cell.
- Always cite `doc_id`, `page`, and `bbox`.
- Prefer values with higher `confidence.structure`.
- If units differ, normalize and show both.
