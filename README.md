# Z-OCR Suite

Z-OCR is a modular OCR-to-RAG toolkit that connects OCR, augmentation, indexing, monitoring, tuning, and reporting. The modern implementation lives in the `zocr/` package (`consensus`, `core`, `orchestrator`), and the legacy single-file `zocr_allinone_merged_plus.py` remains for compatibility.

## Layout
```
zocr/
  consensus/zocr_consensus.py    # OCR + table reconstruction helpers
  core/zocr_core.py              # augmentation, BM25, monitoring, SQL & RAG export
  orchestrator/zocr_pipeline.py  # CLI pipeline orchestrator + resume/watchdog/reporting
  diff/                          # semantic diff engine (see zocr/diff/README.md)
samples/
  demo_inputs/                   # place your PDFs/PNGs here for quick demos
README.md
zocr_allinone_merged_plus.py     # legacy single-file bundle
```

## Quickstart
```bash
# 1) Dependencies
python -m pip install numpy pillow tqdm numba
# For PDFs, install either poppler-utils (pdftoppm) or pypdfium2.

# 2) (Optional) Add sample inputs under samples/demo_inputs/

# 3) Run the pipeline
python -m zocr run --input demo --snapshot --seed 12345

# 4) Resume after a failure
python -m zocr run --outdir out_invoice --resume --seed 12345
```

The legacy entry point `python -m zocr pipeline ...` and the simplified `zocr_allinone_merged_plus.py` remain available for existing workflows.

## Unified CLI
| Command | Description |
|---------|-------------|
| `python -m zocr run …` | Run the end-to-end pipeline (default). |
| `python -m zocr pipeline …` | Alias of `run`, keeps legacy flags. |
| `python -m zocr consensus …` | Consensus/table reconstruction CLI. |
| `python -m zocr core …` | Multi-domain core (augment/index/query/sql/monitor). |
| `python -m zocr simple …` | Lightweight modular OCR using the simple or mock stack. |

## Provider templates and helper models
Generate ready-to-edit JSON presets for downstream LLMs and helper VLMs:
```bash
python - <<'PY'
from zocr.resources.model_provider_presets import PROVIDER_ENV_VARS, write_provider_templates
print("wrote", write_provider_templates("provider_templates.json"))
print("env vars:", ", ".join(sorted(PROVIDER_ENV_VARS)))
PY
```

The JSON contains `downstream_llm` and `helper_vlm` sections. Fill in API keys and model IDs to wire your stack. Supported cloud providers now include:
- OpenAI (`OPENAI_API_KEY`)
- Anthropic (`ANTHROPIC_API_KEY`)
- Google Gemini (`GOOGLE_API_KEY`)
- Azure OpenAI (`AZURE_OPENAI_API_KEY`)
- AWS Bedrock (`AWS_BEDROCK_API_KEY`)
- Mistral (`MISTRAL_API_KEY`)
- xAI (`XAI_API_KEY`)

Helper models can mix and match: e.g., OpenAI `gpt-4o` for vision, Gemini for fast multimodal prompts, or Mistral/xAI for text or vision-capable helpers where supported.

You can prime these environment variables using any secrets manager. The presets keep placeholders identical to the required env var names so `provider_templates.json` can be patched with simple tooling (e.g., `envsubst`).

## Lightweight modular OCR
```bash
# Default simple stack (Tesseract + geometric classification)
python -m zocr simple --images samples/demo_inputs/invoice_page.png --out out_simple.json

# Switch to mocks when avoiding external dependencies
python -m zocr simple --images samples/demo_inputs/invoice_page.png --out out_mock.json --use-mocks
```

PDF inputs fall back to PyMuPDF when Poppler/pdf2image are unavailable. When `DocumentInput.dpi` is not provided, the handler renders PDFs at `default_pdf_dpi` (200 by default).

## Semantic diff
Compare run outputs via:
```bash
python -m zocr.diff --a out/A --b out/B
```
This emits JSON events plus unified-text and HTML reports; see `zocr/diff/README.md` for details.

## RAG integration
The pipeline exports RAG bundles with Markdown digests, cell JSONL, table sections, and suggested queries under `rag/manifest.json`. Suggested feedback actions and hotspot galleries are stored alongside the manifest to help rerun selective fixes.

## Troubleshooting
- `python -m zocr core diagnose --json` surfaces dependency readiness (Poppler, Numba, C extensions).
- PDF rendering respects `ZOCR_PDF_WORKERS`, `ZOCR_PDF_PIXEL_BUDGET`, and related environment variables to cap throughput and DPI.

## Tests
Run the full suite with:
```bash
python -m pytest
```
