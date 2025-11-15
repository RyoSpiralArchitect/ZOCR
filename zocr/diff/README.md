# Semantic Diff Module

The `zocr.diff` package ships a minimal-yet-practical semantic diff engine that runs directly on the
RAG bundle artifacts the pipeline already emits. The module consumes `cells.jsonl` (and optionally
`sections.jsonl`) to align tables/sections between two runs and emit structured events plus multiple
rendering formats for reviewers.

## Directory layout
```
zocr/diff/
 ├─ __init__.py        # public surface
 ├─ differ.py          # table/row/column matching + event generation
 ├─ render.py          # unified-text + HTML renderers
 └─ cli.py             # standalone CLI wrapper
```

## Input expectations
* `cells.jsonl` – required; structured cell records with `page`, `table_index`, `row`, `col`,
  `text`, `filters`, `trace_id`, and optional header metadata.
* `sections.jsonl` – optional; chapter/section headings used when present to produce
  heading-level diff events.

Both files are already exported under `out/<run>/rag/` by the orchestrator, so the diff engine can be
invoked immediately after two runs complete.

## Built-in CLI
```bash
python -m zocr.diff.cli \
  --a out/A/rag/cells.jsonl \
  --b out/B/rag/cells.jsonl \
  --out_diff out/diff/changes.diff \
  --out_json out/diff/events.json \
  --out_html out/diff/report.html
```

Arguments `--sections_a` / `--sections_b` (optional) let the CLI ingest explicit section manifests;
otherwise it will auto-discover `sections.jsonl` next to each `cells.jsonl` when available.

## Why the implementation stays small
The OCR/RAG pipeline already provides nearly every ingredient that a semantic diff system needs:

1. **Structured cell context** – each record carries `page`, `table_index`, `row`, `col`, the text,
   filters (normalized numeric/date metadata), and `trace_id`. Diffing becomes a matter of aligning
   existing structure rather than re-OCRing.
2. **Ready-made RAG bundle** – orchestrator outputs (`cells.jsonl`, `sections.jsonl`, table summaries,
   markdown previews) offer direct A/B comparisons without extra preprocessing.
3. **Trace IDs** – strings such as `doc=A;page=1;table=0;row=3;col=2` encode structural provenance,
   enabling stable row/column tracking even when ordering shifts.
4. **Normalised semantic hints** – `filters.amount`, `filters.qty`, `filters.date`, `synthesis_window`,
   vector fingerprints, etc., let the differ mix text similarity with numeric comparisons and context.
5. **Feedback loop hooks** – monitoring/reanalysis/autotune infrastructure already records changes,
   so diff events can plug directly into the existing observability and escalation mechanisms.
6. **JSONL inputs** – pointing the differ at `out/A/rag/cells.jsonl` and `out/B/rag/cells.jsonl`
   delivers instant semantic diffs with fewer than a couple dozen lines of orchestration code.

Because these pieces exist, the diff engine focuses purely on matching heuristics, event schemas, and
renderers—the pipeline does not need bespoke exporters or bespoke logging to adopt semantic diffs.
