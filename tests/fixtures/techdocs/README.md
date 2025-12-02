# Tech docs visual benchmark fixtures

This folder contains a small benchmark that stresses visual-heavy questions in technical PDFs.
It ships three PDFs alongside a benchmark manifest and sample predictions:

- `machine_schematic.pdf` — wiring diagram with labeled components.
- `control_panel_table.pdf` — configuration table with connector speeds and addresses.
- `safety_flow.pdf` — procedural flow chart for a lockout/tagout scenario.
- `benchmark.json` — list of benchmark questions and expected answers/visual targets.
- `sample_predictions.json` — example prediction payload that satisfies the benchmark and can be
  used for sanity checks.

## Benchmark schema
Each entry in `benchmark.json` includes:

- `id`: unique identifier for the question.
- `pdf`: the PDF filename in this directory.
- `page`: page number containing the relevant visual (1-indexed).
- `query`: user question to run against the document.
- `answers`: acceptable answers (case-insensitive match).
- `visual_type`: one of `diagram`, `table`, or `procedure`.
- `visual_target`: short string describing the object the retriever should surface.
- `answer_requires_visual`: flag indicating the question cannot be answered from text alone.

Predictions are expected as a JSON array with objects like:

```json
{
  "id": "table-comm-baud-rate",
  "predicted_answer": "9600 bps",
  "retrieved_visuals": ["j7 communications row", "diagnostics j5 row"]
}
```

## Running the evaluation
Use the bundled script to score a predictions file:

```bash
python tests/techdocs_eval.py \
  --predictions tests/fixtures/techdocs/sample_predictions.json \
  --benchmark tests/fixtures/techdocs/benchmark.json \
  --top-k 3
```

This prints the top-k visual hit rate for diagram/table questions and end-to-end answer accuracy
for questions that require visual grounding.

## Adding new test PDFs or questions
1. Keep PDFs concise (1–2 pages, minimal file size) so the repository stays lightweight.
2. Add the PDF into this directory and commit it.
3. Append one or more entries to `benchmark.json` with the new `id`, `query`, `answers`, `visual_type`,
   and `visual_target`. Set `answer_requires_visual` to `true` for visual questions.
4. (Optional) Update `sample_predictions.json` with stubbed predictions for the new questions so
   `tests/techdocs_eval.py` can be smoke-tested without a model run.
5. If you regenerate PDFs locally, ensure fonts and drawings remain simple so OCR and layout models
   can parse them consistently.
