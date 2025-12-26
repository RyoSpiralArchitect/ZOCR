# Z-OCR Suite / Z-OCR スイート / Suite Z-OCR

## 概要 / Overview / Aperçu
- The repo focuses on the modular `zocr/` package (`consensus`, `core`, `orchestrator`) that chains OCR → augmentation → indexing → monitoring → tuning → reporting. The legacy `zocr_allinone_merged_plus.py` remains as a drop-in backup.

> Structure-aware RAG blueprint: see [`docs/hybrid_rag_zocr.md`](docs/hybrid_rag_zocr.md) for a minimal design that injects Z-OCR semantic tags into hybrid retrieval.

> **LLM/VLM endpoints:** ready-to-fill templates for downstream LLMs and helper vLMs (local HF path, AWS Bedrock, Azure OpenAI, Gemini, Anthropic) live in [`docs/provider_endpoints.md`](docs/provider_endpoints.md) and [`samples/llm_vlm_endpoints.example.yaml`](samples/llm_vlm_endpoints.example.yaml).

## レイアウト / Layout / Structure
```
zocr/
  consensus/zocr_consensus.py    # OCR + table reconstruction helpers
  core/zocr_core.py              # augmentation, BM25, monitoring, SQL & RAG export
  orchestrator/zocr_pipeline.py  # CLI pipeline orchestrator + resume/watchdog/reporting
  diff/                          # semantic diff engine (see zocr/diff/README.md)
samples/
  demo_inputs/                   # place your PDFs/PNGs here for quick demos
README.md
zocr_allinone_merged_plus.py     # legacy single-file bundle (same features)
```

### Semantic diff overview / セマンティック差分概要 / Aperçu du diff sémantique
- `zocr.diff` compares `cells.jsonl` / `sections.jsonl` to emit JSON events plus unified-text and HTML reports. You can point it at run directories via `python -m zocr.diff --a out/A --b out/B`; see [`zocr/diff/README.md`](zocr/diff/README.md) for layout and CLI details.

## クイックスタート / Quickstart / Démarrage rapide
```bash
# 1. 依存関係 / Dependencies / Dépendances
python -m pip install numpy pillow tqdm numba
# PDF を扱う場合は以下のいずれかを追加:
#   • macOS/Linux: brew/apt 等で poppler-utils (pdftoppm) を入れる
#   • もしくは `python -m pip install pypdfium2`

# 2. サンプル入力（任意） / Optional sample input / Entrée d'exemple optionnelle
#   → Put your PDFs/PNGs under samples/demo_inputs/ (or keep it empty to use the synthetic demo)

# 3. パイプライン実行 / Run the pipeline / Lancer le pipeline
python -m zocr run --input demo --snapshot --seed 12345

# 4. 途中再開 / Resume after failure / Reprendre après un échec
python -m zocr run --outdir out_invoice --resume --seed 12345
```

> `python -m zocr pipeline ...` や `python -m zocr.orchestrator.zocr_pipeline ...` も利用できます。`--domain` を省略するとファイル名と OCR 結果から自動判別されます。
> `python -m zocr pipeline ...` and the legacy `python -m zocr.orchestrator.zocr_pipeline ...` remain available; omitting `--domain` keeps automatic detection.
> `python -m zocr pipeline ...` et l'ancienne forme `python -m zocr.orchestrator.zocr_pipeline ...` restent possibles ; sans `--domain` la détection est automatique.

## 統一 CLI / Unified CLI / Interface unifiée
- `python -m zocr run ...` triggers the orchestrator, while `consensus` and `core` expose the specialised CLIs.

| Command | 説明 / Description / Description |
|---------|----------------------------------|
| `python -m zocr run …` | **JA:** パイプライン実行（デフォルト）。<br>**EN:** Run the end-to-end pipeline (default).<br>**FR:** Lance la chaîne complète (par défaut). |
| `python -m zocr pipeline …` | **JA:** `run` と同義、旧来オプション保持。<br>**EN:** Alias of `run`, keeps legacy flags.<br>**FR:** Alias de `run`, conserve les options historiques. |
| `python -m zocr consensus …` | **JA:** OCR/テーブル復元 CLI（デモ・エクスポート）。<br>**EN:** Consensus/table reconstruction CLI.<br>**FR:** CLI pour la reconstruction de tableaux. |
| `python -m zocr core …` | **JA:** マルチモーダルコア（augment/index/query/sql/monitor）。<br>**EN:** Multi-domain core (augment/index/query/sql/monitor).<br>**FR:** Noyau multi-domaine (augment/index/query/sql/monitor). |
| `python -m zocr simple …` | **JA:** 軽量なモジュラー OCR（シンプル/モック部品）。<br>**EN:** Lightweight modular OCR using the simple or mock stack.<br>**FR:** OCR modulaire léger avec composants simples ou mocks. |

## LLM/vLM provider templates / LLM・vLM プロバイダー雛形 / Gabarits de fournisseurs LLM/vLM
- Emit ready-to-edit JSON presets (split into downstream LLM vs. helper vLM) so you can drop in a local HF model path or cloud credentials (OpenAI, Azure OpenAI, AWS Bedrock, Google Gemini, Anthropic, Mistral, xAI).

```bash
python - <<'PY'
from zocr.resources.model_provider_presets import PROVIDER_ENV_VARS, write_provider_templates

print("wrote", write_provider_templates("provider_templates.json"))
print("env vars:", ", ".join(sorted(PROVIDER_ENV_VARS)))
PY
```

`provider_templates.json` には `downstream_llm` と `helper_vlm` セクションがあり、`model_path` や `api_key` を書き換えるだけで配線できます。
`provider_templates.json` exposes `downstream_llm` and `helper_vlm` sections—just edit `model_path` / `api_key` fields to wire your stack. Environment variables include `OPENAI_API_KEY`, `AZURE_OPENAI_API_KEY`, `AWS_BEDROCK_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, `MISTRAL_API_KEY`, and `XAI_API_KEY`, so you can pre-seed secrets with tools like `envsubst` or your secrets manager.
`provider_templates.json` contient les sections `downstream_llm` et `helper_vlm` ; il suffit de modifier `model_path` / `api_key` pour les relier.

## CLI フラグ / CLI Flags / Options CLI
| Flag | 説明 / Description / Description |
|------|----------------------------------|
| `--input` | **JA:** 入力画像/PDF のパス。`demo` で `samples/demo_inputs/` 配下の実ファイルを一括処理。<br>Paths to images/PDFs; pass `demo` to sweep every real sample under `samples/demo_inputs/`. <br>FR: Chemins vers images/PDF ; `demo` analyse tous les fichiers de `samples/demo_inputs/`. |
| `--outdir` | 出力先 / Output directory / Répertoire de sortie。既定: `out_allinone`. |
| `--dpi` | PDF レンダリング時の DPI / DPI for PDF rendering / DPI pour le rendu PDF. |
| `--domain` | ドメインヒント。未指定または `auto`/`detect` の場合はファイル名・OCR テキストから自動推定（`invoice`, `bank_statement`, `tax` なども指定可）/ Domain hint; leave empty or set to `auto` to infer from filenames + OCR / Indice de domaine ; vide ou `auto` pour une détection à partir des fichiers + OCR. |
| `--k` | BM25 ヒット上位件数 / Top-K for BM25 / Top-K BM25. |
| `--no-tune` | チューニング無効化 / Skip autotune / Désactiver l'auto-réglage. |
| `--tune-budget` | オートチューニング評価回数 / Trials for autotune / Itérations autotune. |
| `--views-log` | ビュー生成ログ CSV を追記するパス / CSV log for microscope/X-ray renders / Journal CSV des rendus microscope/X-ray. |
| `--gt-jsonl` | モニタ評価用のラベル JSONL / Ground-truth JSONL for monitoring / JSONL vérité terrain pour la surveillance. |
| `--org-dict` | 組織名辞書へのパス / Org-name dictionary path / Chemin dictionnaire d'organisations. |
| `--resume` | `pipeline_history.jsonl` を参照して段階をスキップ / Resume stages via `pipeline_history.jsonl` / Reprendre les étapes via `pipeline_history.jsonl`. |
| `--seed` | 乱数シード / RNG seed / Graine aléatoire. |
| `--snapshot` | `pipeline_meta.json` に環境情報を保存 / Persist environment metadata / Conserver les métadonnées d'environnement. |
| `--toy-lite` | **JA:** Toy OCR を軽量化（demo 入力では自動ON）。<br>Clamp toy OCR sweeps + force numeric columns (auto-enabled for `demo`).<br>FR: Allège le Toy OCR (balayages bornés + colonnes numériques forcées, activé automatiquement pour `demo`). |
| `--toy-sweeps` | **JA:** Toy OCR の閾値スイープ上限を明示指定。<br>Explicit upper bound for toy OCR threshold sweeps.<br>FR: Borne supérieure explicite pour les balayages de seuil du Toy OCR. |
| `--autocalib [N]` | **JA:** 表検出の自動キャリブレーションを N ページで実行（値省略時は 3、`0`/未指定で無効）。<br>Auto-calibrate the table detector with N sample pages (defaults to 3 when the flag has no value; pass `0`/omit to skip).<br>FR: Calibre automatiquement le détecteur de tableaux sur N pages (3 si aucune valeur n’est donnée ; `0` ou absence de l’option pour ignorer). |
| `--autotune [N]` | **JA:** 表検出パラメータを N 試行で自動チューニング（値省略時は 6、`0`/未指定で無効）。<br>Run the unsupervised table autotuner for N trials (defaults to 6 without an explicit value; pass `0`/omit to disable).<br>FR: Lance l’autotune non supervisé sur N essais (6 par défaut sans valeur ; `0` ou absence de l’option pour le désactiver). |
| `--force-numeric-by-header` | **JA:** ヘッダ名に応じて数量/単価/金額/税率を数値に正規化。<br>Normalize qty/unit price/amount/tax columns according to headers.<br>FR: Normalise les colonnes quantitatives selon les en-têtes. |
| `--ingest-signature` | **JA:** 別環境での再現ログ（signature JSON）を読み込み差分チェック。<br>Ingest reproducibility signature JSON from another run to compare diffs.<br>FR: Ingère une signature de reproductibilité externe pour comparer les écarts. |
| `--advisor-response` | **JA:** 外部アドバイザ（LLM等）の助言ファイルを与えて再解析/監視の再実行に接続。<br>Feed advisor (LLM) responses so the orchestrator can trigger reruns based on the advice.<br>FR: Fournit une réponse d’advisor afin de relancer réanalyse/monitoring selon les recommandations. |
| `--tess-unicharset` | **JA:** Toy OCR の文字集合のみ差し替え。辞書・バイグラムは常に `zocr.resources.domain_dictionary` を使用。<br>Override the toy OCR glyph set; the lexicon/bigram tables always come from the bundled `zocr.resources.domain_dictionary`. <br>FR: Remplace uniquement l'ensemble de glyphes du Toy OCR ; le dictionnaire et les bigrammes proviennent de `zocr.resources.domain_dictionary`. |

> **[JA]** `python -m zocr run/pipeline …` や `python -m zocr consensus …`、`zocr_allinone_merged_plus.py` では `--autocalib [N]` / `--autotune [N]` を使うと、フラグ単体で既定回数（それぞれ 3 / 6）を実行し、`0` または未指定で無効化できます。<br>**[EN]** The orchestrator (`python -m zocr run` / `pipeline`), the standalone consensus CLI, and the legacy bundle all share the optional `--autocalib [N]` and `--autotune [N]` switches: invoking a flag without a value runs the default sample/trial counts (3 / 6) while omitting it or passing `0` keeps the pass disabled.<br>**[FR]** L’orchestrateur (`python -m zocr run` / `pipeline`), la CLI `zocr consensus` et le bundle monolithique partagent désormais `--autocalib [N]` / `--autotune [N]` : appeler l’option sans valeur lance les passes par défaut (3 / 6), alors qu’une valeur `0` ou l’absence de l’option les désactive.

## サブコマンド / Subcommands / Sous-commandes
- `history --outdir out_invoice --limit 10` — 直近の処理履歴を表示 / show recent history / affiche l'historique récent。
- `summary --outdir out_invoice --keys sql_csv rag_manifest` — 生成物を JSON 出力 / print artifacts / affiche les artefacts。
- `plugins [--stage post_rag]` — 登録済みプラグインを列挙 / list registered hooks / lister les hooks enregistrés。
- `report --outdir out_invoice --open` — 三言語 HTML ダッシュボード生成 / build trilingual HTML dashboard / générer un tableau de bord HTML trilingue。
- `diagnose [--json]` — 依存関係の自己診断（Poppler/Numba/C拡張など）/ dependency self-check for Poppler/Numba/C helpers / autodiagnostic des dépendances (Poppler/Numba/extensions C).

### 軽量モジュラー OCR / Lightweight modular OCR / OCR modulaire léger
```bash
# 本番デフォルトのシンプル構成（Tesseract + 幾何学ベースの分類）
python -m zocr simple --images samples/demo_inputs/invoice_page.png --out out_simple.json

# 依存関係を避けたいときはモック部品に切り替え可能
python -m zocr simple --images samples/demo_inputs/invoice_page.png --out out_mock.json --use-mocks
```

## 仕組み / Mechanics / Fonctionnement
1. **OCR & Consensus** — `zocr.consensus.zocr_consensus` がレイアウト解析とセル信頼度計算を実行。
2. **Export JSONL** — RAG に適した JSONL を出力し、`pipeline_history.jsonl` に記録。
3. **Augment & Index** — `zocr.core.zocr_core` が多領域特徴、BM25、SQL スナップショット、RAG バンドルを構築。
4. **Monitor & Tune** — ヒット率、p95 レイテンシ、失敗率を監視し、必要に応じて自動調整後に再監視。
5. **Report & Plugins** — HTML レポート、要約 JSON、RAG マニフェスト、プラグインフック（`post_export`/`post_index`/`post_monitor`/`post_sql`/`post_rag`）を呼び出し。

各段階は `_safe_step` でガードされ、成功・失敗・経過時間を `pipeline_history.jsonl` に追記します。

- **[EN]** The schema rectifier now merges the first rows into synthetic headers before falling back to semantic/heuristic detection and logs the winning header source plus strategy breakdown under `schema_alignment`, so headerless Japanese invoices/estimates still align automatically.

- **[EN]** During export we reuse the consensus `row_bands` so OCR crops stay aligned with the reconstructed rows.
- **[EN]** The toy OCR now inspects edge brightness to auto-detect inverted text (white-on-black) and cuts down low-confidence cells.
- **[EN]** The toy OCR persists its glyph atlas and N-gram priors to `toy_memory.json` (override via `ZOCR_TOY_MEMORY`) so previously seen glyphs and vocabulary stay available. During a run it now keeps low-confidence patches in a short-term store and replays them whenever a new glyph variant is learned; tune the cache and queue sizes via `ZOCR_GLYPH_CACHE_LIMIT` / `ZOCR_GLYPH_PENDING_LIMIT`. The N-gram model applies an exponential moving average controlled by `ZOCR_NGRAM_EMA_ALPHA` to balance retention and forgetting, and the `toy_memory` summary still lists load/save/delta snapshots while surfacing runtime cache hits, replay gains, and surprisal metrics. The `toy_memory_versioned/summary.json` timeline now records per-epoch deltas together with aggregated `stats` (epoch count, growth averages, stagnant streaks, recent surprisal/runtime replay ratios) and a trimmed `tail` digest so operators can review momentum without opening each epoch file. The intent engine inspects those memory changes plus recognition stats to trigger reanalysis or header focus automatically while `intent_simulations` projects the impact of adjusting `ocr_min_conf` and `lambda_shape`.
- **[EN]** The exporter now tracks N-gram surprisal alongside confidence and queues even high-confidence cells for learning when they cross the `ZOCR_SURPRISAL_REVIEW_THRESHOLD`; the `*.signals.json` payload reports both low-confidence and high-surprisal counts. The intent engine inspects `high_surprisal_ratio` so a spike in contextually unlikely cells triggers an immediate reanalysis pass within the same run.
- **[EN]** The reanalysis stage (`reanalyze_learning_jsonl`) still calls into Tesseract when present, but the synthetic fallback now sweeps adaptive thresholds, performs word segmentation, posterizes, and sharpens aggressively to emulate Tesseract-style outputs when the engine is missing. The ambiguity map continues to remap noisy glyphs such as `??I` → `771`, and the summary exposes both `external_engines` counts and the fallback breakdown via `fallback_transform_usage` / `fallback_variant_count`.
- **[EN]** A hotspot analyzer now scans `*.learning.jsonl` to populate `learning_hotspots` plus a `selective_reanalysis_plan`; reanalysis limits itself to the traced rows/columns so huge tables no longer trigger a full sweep. The `reanalyze_learning_jsonl` summary records the applied `focus_plan` and `focus_stats` so you can see how many cells were targeted.
- **[EN]** Reanalysis outputs now feed straight back into `doc.contextual.reanalyzed.jsonl` (or rewrite the original in place), updating the paired `*.signals.json` with an `applied_reanalysis` block that tracks improved counts and average deltas. The pipeline summary records each pass under `reanalysis_applied`, and downstream augment/index/monitor stages consume the refreshed text automatically.
- **[EN]** The retrieval layer now blends BM25 + keyword + image similarity with a symbolic scorer that inspects the structured `filters`, improving Trust@K for downstream RAG agents.

## 自動ドメイン検出 / Automatic Domain Detection / Détection automatique du domaine
- **[EN]** The orchestrator mines folder/file tokens, maps them through `DOMAIN_KW` and `_DOMAIN_ALIAS`, then refines the guess by scanning the exported JSONL. Confidence scores must clear a 0.25 threshold before overriding prior hints; the full trace lives in `pipeline_summary.json` under `domain_autodetect`.

## 生成物 / Outputs / Résultats
- `doc.zocr.json` — OCR & consensus の主 JSON。
- `doc.mm.jsonl` — マルチモーダル JSONL（RAG / BM25 共用）。
- `rag/` — `export_rag_bundle` によるセル/テーブル/Markdown/マニフェスト。
- `rag/trace.prov.jsonld` — PROV-O 互換の系譜バンドル。`doc.mm.jsonl` → `rag/cells.jsonl` → `sections` / `tables` → `manifest` の派生関係と生成アクティビティを追跡できます。
- **[JA/EN]** `python -m zocr.core embed --jsonl rag/cells.jsonl --model <path>` で (EC2 で同期した SentenceTransformer などの) 埋め込み
  を `.embedded.jsonl` に付与できます。Bedrock など AWS サービス経由なら `--provider bedrock --model <modelId> --aws-region <region>`
  で同じ JSONL にベクトルを付与できます / Attach embeddings from a local SentenceTransformer (e.g., your EC2-resynced model)
  with `.embedded.jsonl` output for downstream RAG; switch to `--provider bedrock --model <modelId> --aws-region <region>` to call AWS services.
- `agentic_requests.json` — Agentic RAG 用の diff 依頼バンドル（差分画像/説明向けプロンプト） / Agentic RAG request bundle for diff overlays + narratives / Bundle Agentic RAG (prompts pour images diff & explications).
- `sql/` — `sql_export` で生成される CSV とスキーマ（`trace` 列で doc/page/table/row/col を Excel から参照可能）。
- `views/` — マイクロスコープ 4 分割＋X-Ray オーバーレイ。
- `reanalyze/` — 低信頼セルの再解析 JSONL（`*.summary.json` には `external_engines` と `fallback_transform_usage` / `fallback_variant_count` を含む詳細統計） / Reanalysis JSONL for low-confidence cells (the accompanying `*.summary.json` captures `external_engines` plus `fallback_transform_usage` / `fallback_variant_count`) / Ré-analyses JSONL des cellules peu fiables (le `*.summary.json` expose `external_engines` ainsi que `fallback_transform_usage` / `fallback_variant_count`).
- `toy_memory.json` — Toy OCR の記憶スナップショット（グリフアトラスと N-gram）。`ZOCR_TOY_MEMORY` 環境変数で場所を固定可能。
- `toy_memory.json` — Snapshot of the toy OCR memory (glyph atlas + N-grams). Set `ZOCR_TOY_MEMORY` to pin the storage path.
- `toy_memory.json` — Instantané de la mémoire du toy OCR (atlas de glyphes + N-grammes). Utilisez `ZOCR_TOY_MEMORY` pour fixer l’emplacement.
- `toy_memory_versioned/` — エポック番号・差分・認識統計を蓄積する履歴ディレクトリ（`summary.json` 付き）。
- `toy_memory_versioned/` — History directory with epoch-stamped payloads and a `summary.json` collecting deltas and recognition stats.
- `toy_memory_versioned/` — Répertoire d’historique horodaté par époque avec `summary.json` listant deltas et statistiques de reconnaissance.
- `pipeline_summary.json` — すべての成果物と依存診断をまとめた要約（`rag_*`, `sql_*`, `views`, `dependencies`, `report_path` など）。
- `rag_trace_schema`, `rag_fact_tag_example` — サマリー内で RAG トレーサの仕様と `<fact ...>` タグ例を公開。
- `monitor.csv` — UTF-8 (BOM 付き) で出力し、Excel/Numbers でも文字化けなく開けます。
- `pipeline_meta.json` — `--snapshot` 有効時の環境情報。
- `pipeline_report.html` — trilingual ダッシュボード（`report` サブコマンドでも再生成可）。
- `episodes/<ID>/` — 各実行のスナップショットを格納（`pipeline_summary.json` / `monitor.csv` / `pipeline_history.jsonl` / `auto_profile.json` / `rag/manifest.json` / `repro_signature.json` / `stage_trace.json` などを丸ごと複製し、`learning_hotspots` や `hotspot_gallery` も JSON 化）。`pipeline_history.jsonl` の各行には `episode_id` が付与され、`episodes_index.json` でドメイン・Hit@K・p95・ゲート結果を一覧できます。

## モニタリング洞察 / Monitoring Insights / Analyse de la surveillance
- `pipeline_summary.json` の `insights` は構造・ゲート・プロファイルの3本立てで、over/under・TEDS・行外れ率や Hit@K を数値付きで提示します。
- `pipeline_summary.json` には `stage_trace` / `stage_stats` も追加され、各 `_safe_step` の経過時間・成功可否・代表的な出力を一覧できます（単体スクリプト版でも同様）。
- **[EN]** `pipeline_summary.json` now ships with `stage_trace` / `stage_stats`, exposing every `_safe_step` duration, status, and a compact output preview (mirrored in the single-file runner).
- `--print-stage-trace` または `ZOCR_STAGE_TRACE_CONSOLE=1` で実行直後にタイミング表を標準出力へ表示できます（遅延ステージや失敗箇所の即時可視化に便利）。
- **[EN]** Use `--print-stage-trace` or set `ZOCR_STAGE_TRACE_CONSOLE=1` to dump the formatted stage timing table to stdout right after a run, making bottlenecks/failures obvious without opening the JSON summary.
- `profile_guard` ブロックには、1ランで変更されたプロファイル項目（既定で最大3件）と、ガードによって拒否/調整されたオーバーライドが記録されます。`ZOCR_PROFILE_MAX_CHANGES` 環境変数で許可数を調整できます。
- `safety_flags.gate_fail_streak` は連続ゲート失敗数を数え、閾値 (`ZOCR_GATE_FAIL_ESCALATE`, 既定3) に到達すると `escalate_to_human` を推奨します。値は `auto_profile.json` にも保存され、次回の run で継続されます。
- **[EN]** A meta-intent layer now narrates why an action was chosen and which hotspots it targets; `pipeline_summary.json` and the generated `rag/feedback_request.*` expose the `meta_intent` story plus its `focus_plan`, giving downstream agents a rationale to follow.
- **[EN]** The orchestrator now crops `learning_hotspots` into `rag/hotspots/*.png`; every gallery entry carries its inferred role (header/body/footer), the ranked reason, and before/after text so reviewers can see the drift without opening JSON. The `hotspot_gallery` block is stored in `pipeline_summary.json` and the RAG request. Tune the export count via `ZOCR_HOTSPOT_GALLERY_LIMIT` (default 12).
- **[EN]** A Markdown companion (`rag/hotspots/gallery.md`) now accompanies the PNG crops so advisors can skim every hotspot with its location, observed text, and reasons without opening the JSON.
- インボイス系ドメインは金額 (`hit_amount>=0.8`) と日付 (`hit_date>=0.5`) の双方が揃わない限り PASS しません。欠損時はゲートが FAIL となり、`gate_reason` で要因を特定できます。
- **[EN]** `monitor.csv` now records `trust_amount`, `trust_date`, and `trust_mean`, exposing how many Top-K hits carry proper provenance. Coverage counters (`tax_coverage`, `corporate_coverage`) clarify when rates are zero because no candidates were found.
- Intent 指向のフィードバックでは `intent.action="reanalyze_cells"` が検知されると同一実行内で再解析フローを即時発火し、結果は `intent_runs` と `learning_reanalyzed_jsonl` に反映されます。
- When the intent engine requests `reanalyze_cells`, the orchestrator now fires a same-run reanalysis pass and records the action under `intent_runs` together with the refreshed `learning_reanalyzed_jsonl` path.
- Lorsqu’un intent `reanalyze_cells` est produit, l’orchestrateur déclenche immédiatement la réanalyse et consigne l’action dans `intent_runs` ainsi que le nouveau `learning_reanalyzed_jsonl`.
- Intentシグナルには `intent_simulations` が追加され、Toyメモリの差分から推定した再解析効果や `ocr_min_conf` / `lambda_shape` を上下させた場合の低信頼率・p95予測を提示します。
- The intent payload now includes `intent_simulations`, providing what-if predictions for lowering/raising `ocr_min_conf` and `lambda_shape` plus the expected impact of reanalysis derived from toy-memory deltas.
- Le résumé d’intent expose désormais `intent_simulations`, des scénarios « et si » pour modifier `ocr_min_conf` / `lambda_shape` et l’effet anticipé d’une réanalyse calculé à partir des deltas de mémoire du toy OCR.
- autotune / `learn_from_monitor` が更新した `w_kw` / `w_img` / `ocr_min_conf` / `lambda_shape` を拾い、ヘッダ補完や再走査の微調整ヒントを返します。

## Toy OCR ランタイムノブ / Toy OCR runtime knobs / Commandes Toy OCR
- **[EN]** Bound threshold sweeps via `ZOCR_TOY_SWEEPS` (default 5, auto-clamped to ~2–4 in toy-lite/demo runs) and opt out of header-driven numeric coercion with `ZOCR_FORCE_NUMERIC=0`.
- `ZOCR_TOY_MEMORY` で Toy OCR のメモリ保存先を固定でき、`ZOCR_GLYPH_CACHE_LIMIT` / `ZOCR_GLYPH_PENDING_LIMIT` / `ZOCR_NGRAM_EMA_ALPHA` がキャッシュ容量や忘却率を制御します。
- `ZOCR_TESS_UNICHARSET` を指定するとグリフ集合のみを差し替えられます。辞書と n-gram は `zocr.resources.domain_dictionary` がまとめたサンプル対応キーワードで常時バンドルされるため、旧来の `--tess-wordlist` / `--tess-bigram-json` は廃止しました。
- **[EN]** Use `ZOCR_TESS_UNICHARSET` (or `--tess-unicharset`) if you need custom glyph coverage; the lexicon/bigram data now ships inside the repo via `zocr.resources.domain_dictionary`, so the legacy `--tess-wordlist` / `--tess-bigram-json` switches are gone.
- **[EN]** The same bundled dictionary powers the toy OCR, consensus exporter, and the `zocr.core` retrieval boosts, so no part of the stack expects external wordlists/bigram JSON anymore.
- **[EN]** If you still need a private wordlist, point `ZOCR_TESS_EXTRA_DICT` to one or more newline-delimited files (use your OS path separator to list multiples) and the toy/tesslite dictionary + bigrams will merge them in, seeding the toy N-gram memory with the same lexicon.
- `ZOCR_TESS_DOMAIN` または CLI の `--domain` / パイプラインの domain 設定を指定すると、Toy OCR の内蔵辞書が該当ドメインのキーワード集合に切り替わります。プロファイルや自動判別で domain が確定するとパイプライン側で `ZOCR_TESS_DOMAIN` も自動更新されます。

## PDF レンダリング最適化 / PDF rasterization knobs / Optimisations PDF
- **[EN]** When Poppler (`pdftoppm`) is missing, the pipeline now falls back to `pypdfium2` automatically, so PDFs can be rasterized without any system packages. If both are available Poppler is used first, with pdfium acting as the safety net.
- **[EN]** The dependency diagnostics (`dependencies.pdf_raster`) now spell out which backend is active (Poppler vs `pypdfium2`), so the summary/logs stop nagging about missing Poppler once the pdfium fallback is installed. The per-backend blocks are also mirrored to `dependencies.poppler_pdftoppm` / `dependencies.pypdfium2`, keeping legacy dashboards and quick JSON viewers in sync.
- **[EN]** For PDFs with ≥6 pages the pdfium path renders pages in parallel (up to four workers by default, auto-tuned to your CPU) which dramatically shortens the raster stage.
- `ZOCR_PDF_WORKERS` を設定するとワーカー数を固定できます（例: `ZOCR_PDF_WORKERS=2 python -m zocr run ...`）。`ZOCR_PDF_PARALLEL_MIN_PAGES` で並列化を開始する閾値も調整可能です。
- Set `ZOCR_PDF_WORKERS` to clamp the worker count (e.g. `ZOCR_PDF_WORKERS=2 python -m zocr run ...`). Use `ZOCR_PDF_PARALLEL_MIN_PAGES` to raise/lower the page-count threshold.
- Fixez `ZOCR_PDF_WORKERS` pour imposer un nombre précis de workers (ex. `ZOCR_PDF_WORKERS=2 python -m zocr run ...`). Le seuil d’activation peut être ajusté via `ZOCR_PDF_PARALLEL_MIN_PAGES`.
- **[EN]** For extremely long or highly visual PDFs the rasterizer now lowers the DPI when the projected pixel budget would explode, cutting down the PNG explosion without sacrificing table fidelity.
- `ZOCR_PDF_PIXEL_BUDGET` でピクセル上限（既定 3.2e8）を、`ZOCR_PDF_MIN_DPI` で自動縮小時の下限 DPI（既定 120）を変更できます。`ZOCR_PDF_MAX_PAGES` を設定するとレンダリングするページ数そのものを頭打ちでき、`ZOCR_PDF_INSPECT_PAGES` はページサイズ見積りに使うサンプル枚数を制御します。
- Tune the limits via `ZOCR_PDF_PIXEL_BUDGET` (default 3.2e8 pixels) and `ZOCR_PDF_MIN_DPI` (default 120 DPI when throttling kicks in). Set `ZOCR_PDF_MAX_PAGES` to hard-cap the number of rendered pages, and adjust the sampling window with `ZOCR_PDF_INSPECT_PAGES` if you want to inspect more/less pages before estimating sizes.
- Ajustez `ZOCR_PDF_PIXEL_BUDGET` (3,2e8 pixels par défaut) et `ZOCR_PDF_MIN_DPI` (120 DPI mini lorsque la réduction s’active). `ZOCR_PDF_MAX_PAGES` plafonne le nombre de pages rasterisées et `ZOCR_PDF_INSPECT_PAGES` contrôle combien de pages sont échantillonnées pour estimer les dimensions.
- **[EN]** Use `ZOCR_PDF_MIN_DPI_FLOOR` (default 72) to define the absolute minimum DPI so the rasterizer can still respect the pixel budget even when the soft minimum is higher.
- **[EN]** When you run with `--snapshot` the orchestrator now sets `ZOCR_PIPELINE_SNAPSHOT=1`, enabling snapshot-specific knobs: `ZOCR_PDF_SNAPSHOT_DPI_PCT` (default 80 %), `ZOCR_PDF_SNAPSHOT_PIXEL_BUDGET` (default 2.2e8), and `ZOCR_PDF_SNAPSHOT_MAX_PAGES` (disabled by default) to throttle DPI, total pixels, or page count just for traced runs.
- **[EN]** Set `ZOCR_TESS_DOMAIN` (or pass `--domain` to the consensus CLI / orchestrator) to clamp the bundled lexicon to a specific domain keyword set. The pipeline writes this env var automatically whenever its profile or autodetector selects a domain.
- **[JA]** 追加指定なしでもリポジトリ同梱の tesslite セット（JP/EN インボイス語彙）が自動で読み込まれます。`--tess-*` / `ZOCR_TESS_*` を指定すると上書きされ、`ZOCR_TESSLITE_DISABLE_BUILTIN=1` で無効化できます。<br>**[EN]** A bundled tesslite glyph/dictionary set now loads automatically—override it with `--tess-*` / `ZOCR_TESS_*` or disable via `ZOCR_TESSLITE_DISABLE_BUILTIN=1`. <br>**[FR]** Un jeu tesslite intégré est actif par défaut ; remplacez-le via `--tess-*` / `ZOCR_TESS_*` ou désactivez-le avec `ZOCR_TESSLITE_DISABLE_BUILTIN=1`.
- **[JA]** motion prior / tesslite / lexical & numeric confidence boost / N-gram EMA / hotspot 検出 / view 生成 / intent simulations はすべて既定でオンになり、`pipeline_summary.json` / `toy_feature_defaults` に適用状況が記録されます。追加の環境変数なしで Toy エンジンのフル機能が動作し、必要に応じてサマリで確認可能です。<br>**[EN]** Motion priors, tesslite dictionaries, lexical & numeric confidence boosts, the N-gram EMA, hotspot detection, microscope/X-ray view generation, and intent simulations are now enabled by default—no extra env vars needed—and the `toy_feature_defaults` block in `pipeline_summary.json` records which knobs were applied. <br>**[FR]** Les motion priors, dictionnaires tesslite, boosts lexical/numérique, EMA des N-grammes, détection de hotspots, vues microscope/X-ray et simulations d’intent sont tous actifs par défaut sans variables d’environnement supplémentaires ; le bloc `toy_feature_defaults` du `pipeline_summary.json` consigne l’état de chaque fonction.
- `--toy-lite` または demo 入力では数値列の強制正規化と sweep クランプが既定で有効になり、`pipeline_summary.json` の `toy_runtime_config` と `last_export_stats` に適用結果が保存されます。
- `--autocalib` / `--autotune` フラグを有効にすると、事前の表キャリブレーション / オートチューニング結果が `pipeline_summary.json` の `table_autocalib` / `table_autotune` として記録され、採用されたパラメータは `table_params` に保存されます。<br>**[EN]** When the optional `--autocalib` / `--autotune` flags are used, the pre-export calibration/tuning summaries are written to `pipeline_summary.json` (`table_autocalib` / `table_autotune`) and the applied values land under `table_params`.<br>**[FR]** Lorsque vous activez `--autocalib` / `--autotune`, les résultats de calibration/auto-réglage sont consignés dans `pipeline_summary.json` (`table_autocalib` / `table_autotune`) et les paramètres retenus figurent dans `table_params`.

## Export 進捗と高速化 / Export progress & acceleration / Export : progression et accélérations
- `ZOCR_EXPORT_OCR` で Export 内の OCR バックエンドを切り替えられます（例: `fast` でセル OCR をスキップし構造のみ書き出し、`toy` / `tesseract` で再解析）。
- **[JA]** `zocr consensus export` では画像フォルダ／ワイルドカード／PDF パスをそのまま渡しても自動でビットマップを列挙し、PDF の場合は Poppler/pdfium でその場レンダリングします。パスが壊れてページ画像を開けなかった場合は `missing page bitmaps` 警告に候補とページ番号をまとめて表示するため、`doc.contextual.jsonl` が空のままでも原因を即座に特定できます。<br>**[EN]** The standalone `zocr consensus export` command now walks directories, glob patterns, or even raw PDF paths for you—PDFs are rasterised on demand via Poppler/pdfium—and it logs `missing page bitmaps` with the attempted candidates so broken paths can be fixed before the JSONL ends up empty.<br>**[FR]** La commande autonome `zocr consensus export` accepte désormais dossiers, motifs glob ou PDF bruts : les PDF sont rasterisés à la volée (Poppler/pdfium) et chaque page introuvable est signalée (`missing page bitmaps`) avec les chemins testés pour éviter des JSONL vides sans diagnostic.
- **[EN]** The per-table export guard (`ZOCR_EXPORT_GUARD_MS` / `--export-guard-ms`) is now opt-in. Leave it unset to let slow tables finish; if you do set a limit the exporter still writes whatever cells were processed and tags them with `guard_timeout` for downstream review.
- Export 時の行バンド再シード（motion prior）は既定で常時有効になりました。`--no-motion-prior` や `ZOCR_EXPORT_MOTION_PRIOR=0` で無効化できます。<br>**[EN]** Motion-prior reseeding between toy export sweeps is now enabled by default; opt out via `--no-motion-prior` or `ZOCR_EXPORT_MOTION_PRIOR=0`. <br>**[FR]** Le motion prior est désormais actif par défaut ; utilisez `--no-motion-prior` ou `ZOCR_EXPORT_MOTION_PRIOR=0` pour revenir à l’exploration exhaustive.
- `ZOCR_EXPORT_PROGRESS=1` と `ZOCR_EXPORT_LOG_EVERY=100`（任意）でセル処理数の進捗ログを標準出力に流し、長大なグリッドでも固まって見えません。
- `ZOCR_EXPORT_MAX_CELLS` を指定すると巨大テーブルをサンプリングできます。進捗ログ有効時は `last_export_stats()` / `pipeline_summary.json` にページ数・セル数・数値強制件数・処理秒数が残ります。
- Toy OCR と組み合わせる場合は `ZOCR_TOY_SWEEPS=2 ZOCR_EXPORT_OCR=fast` で 4 ページ超のインボイスでも即時に JSONL/SQL/RAG を生成できます。

## アドバイザ/再現性ループ / Advisor & reproducibility loops / Boucles advisor + reproductibilité
- **[EN]** Supply `--advisor-response advisor.jsonl` (JSON or plaintext) to ingest LLM/agent feedback; parsed actions such as `reanalyze_cells` or `rerun_monitor` feed directly into the rerun controller and their effects are logged under `advisor_ingest` / `advisor_applied`.
- 再現署名は `pipeline_summary.json` の `repro_signature` に保存され、外部共有→`--ingest-signature` で読み込むことで「期待値との差分」を自動で可視化できます。

## 対応ドメイン / Supported Domains / Domaines pris en charge
- インボイス (JP/EN/FR)、見積書、納品書、領収書、契約書、購入注文書、経費精算、タイムシート、出荷案内、銀行明細、公共料金請求書。
- 医療領収書に加え **医療請求 (medical_bill)**、**通関申告 (customs_declaration)**、**助成金申請 (grant_application)**、**搭乗券 (boarding_pass)**、**銀行明細 (bank_statement)**、**公共料金 (utility_bill)**、**保険金請求 (insurance_claim)**、**税務申告 (tax_form)**、**給与明細 (payslip)**、**出荷案内 (shipping_notice)**、**経費精算 (expense_report)** を新たに追加。既存の **賃貸借契約**、**ローン明細**、**旅行行程** も強化済みです。
- 各ドメインは `DOMAIN_KW`/`DOMAIN_DEFAULTS`/`DOMAIN_SUGGESTED_QUERIES` を共有し、RAG 推奨クエリやウォームアップ検索を自動設定します。
- **[EN]** Invoice defaults now use an `ocr_min_conf` around 0.55 so faint print and pastel backgrounds stay captured.

## RAG 連携 / RAG Integration / Intégration RAG
- `export_rag_bundle` が Markdown ダイジェスト、セル JSONL、テーブル別セクション、推奨クエリを `rag/manifest.json` にまとめます。
- サマリー (`summary` サブコマンド) から `rag_markdown` や `rag_suggested_queries` を取得し、下流エージェントに渡せます。
- `post_rag` フックでカスタム転送やストレージ連携を差し込めます。
- **Embedding / 埋め込み** — RAG マニフェストには既定で AWS Bedrock Titan の埋め込みプラン（`amazon.titan-embed-text-v2`）と
  `embedding.hint` を同梱し、`aws bedrock-runtime invoke-model --model-id amazon.titan-embed-text-v2 --body '{"inputText": "hello world"}'`
  などの CLI 例や Python スニペットを添付します。地域や制約に合わせてモデル名だけ差し替えれば、そのまま Bedrock へ投げられます。
- **[EN]** Each cell now ships with a `trace` string (`doc=...;page=...;row=...`) plus an immutable `<fact trace="...">…</fact>` tag so RAG/LLM stacks can demand provenance; see `rag_trace_schema` and `rag_fact_tag_example` in the summary.
- **Feedback loop / フィードバックループ** — `rag/manifest.json` に `"feedback": {"profile_overrides": {...}, "actions": ["reanalyze_cells"], "notes": "..."}` を追記すると、次回 `--resume` 実行時に自動で読み込みます。`auto_profile.json` へ上書きし、`actions` には `reanalyze_cells` / `rerun_monitor` / `rerun_augment` などを記載できます。
- `--rag-feedback path/to/manifest.json` を指定すると、別ディレクトリで生成した RAG マニフェストでも同じ処理（プロファイル上書き + rerun 指示）を適用できます。適用結果は `pipeline_summary.json` の `rag_feedback`, `rag_feedback_actions(_applied)` に記録されます。
- `rag/feedback_request.json` / `.md` が各実行後に生成され、現在のメトリクス・意図・進捗と共に「manifest の feedback ブロックをどのように編集すべきか」を JA/EN のガイド付きで提示します。LLM や外部エージェントに渡せば、そのまま `feedback` を追記して `--resume` で再投入できます。
- 生成されるフィードバックリクエストには `meta_intent`・`learning_hotspots`・`selective_reanalysis_plan` も含まれるため、外部 LLM が「どのセルを直せばよいか」を即座に把握できます。
- The feedback bundle now embeds the latest `meta_intent`, `learning_hotspots`, and `selective_reanalysis_plan`, so external LLMs can reason about root causes and suggest precise fixes instead of scanning the entire run.
- Le paquet de feedback inclut désormais `meta_intent`, `learning_hotspots` et `selective_reanalysis_plan`, ce qui permet à un LLM externe d’identifier immédiatement les zones à corriger.
- **[EN]** The `rag/feedback_request.*` bundle now ships with reviewer questions tailored to the current hotspots, meta-intent story, and low-confidence metrics, so a human or LLM can answer them directly before pasting the response back into the manifest.
- `hotspot_gallery` (JSON + `rag/hotspots/*.png`) is also exported so advisors can see the offending cells without reopening the PDFs; disable or resize the gallery via `ZOCR_HOTSPOT_GALLERY_LIMIT`.
- **[EN]** Every run now appends to `rag/conversation.jsonl`, logging the emitted feedback requests plus any ingested manifest/advisor responses. `pipeline_summary.json` exposes the path + last entry under `rag_conversation`, so external agents can replay or extend the dialog.

## ビジュアライゼーション / Visualisation / Visualisation
- 4 パネルのマイクロスコープ（原画、シャープ、二値、勾配）と X-Ray オーバーレイを自動生成。
- `--views-log` で生成履歴を CSV 追記し、監視や QA に利用可能。

## サンプル素材 / Sample Assets / Ressources d'exemple
- `samples/demo_inputs/` にファイルを配置すると、`--input demo` がそれらを取り込みます。
- フォルダは空でも構いません。クローン直後は同梱の合成サンプルが利用されます。
- `samples/README.md` に多言語で手順をまとめています。
- 旧来の `samples/<domain>/` フォルダは廃止し、`samples/demo_inputs/` に一本化しました。各ドメインのキーワードや `--domain` 推奨値は `samples/README.md` 内の早見表に集約しています。

## 依存関係 / Dependencies / Dépendances
- Python 3.9+
- NumPy, Pillow, tqdm（コア機能）
- Numba（任意、BM25 DF の並列化）
- Poppler など PDF レンダラ（PDF 入力時に必要な場合あり）

## バックアップの単一ファイル版 / Single-file Backup / Fichier unique de secours
- 既存ワークフローが `zocr_allinone_merged_plus.py` に依存している場合、そのまま利用できます。
- モジュラー版と同じ CLI/サブコマンド/プラグイン/レポート機能を保持しています。
