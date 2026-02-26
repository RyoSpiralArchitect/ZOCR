# Z-OCR Suite / Z-OCR スイート / Suite Z-OCR

## 概要 / Overview / Aperçu
- **[JA]** `zocr/` パッケージ（`consensus`, `core`, `orchestrator`）を中心に、OCR→拡張→インデックス→監視→調整→レポートまでをワンコマンドで連結します。
- **[EN]** The repo focuses on the modular `zocr/` package (`consensus`, `core`, `orchestrator`) that chains OCR → augmentation → indexing → monitoring → tuning → reporting.
- **[FR]** Le paquet modulaire `zocr/` (`consensus`, `core`, `orchestrator`) relie OCR → augmentation → indexation → surveillance → réglage → rapport via une seule commande.

## ライセンス / License / Licence

- **[JA]** 本プロジェクトは **AGPL-3.0-or-later** と **商用ライセンス** のデュアルライセンスです。詳細: `LICENSING.md`
- **[EN]** This project is dual-licensed under **AGPL-3.0-or-later** and a **commercial license**. See: `LICENSING.md`
- **[FR]** Ce projet est en double licence (**AGPL-3.0-or-later** + licence commerciale). Voir : `LICENSING.md`

## 引用 / Citation / Citation

- **[JA]** 学術用途などで引用する場合は `CITATION.cff` を参照してください。変更履歴: `CHANGELOG.md`
- **[EN]** For academic citation, see `CITATION.cff`. Release notes: `CHANGELOG.md`
- **[FR]** Pour citer ce logiciel, voir `CITATION.cff`. Historique: `CHANGELOG.md`

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
```

### Semantic diff overview / セマンティック差分概要 / Aperçu du diff sémantique
- **[JA]** `zocr.diff` は `cells.jsonl` / `sections.jsonl` を比較してイベントJSON・.diff・HTMLを生成します。`python -m zocr.diff --a out/A --b out/B` のように実行ディレクトリを渡せます。構成や詳細は [`zocr/diff/README.md`](zocr/diff/README.md) を参照してください。
- **[EN]** `zocr.diff` compares `cells.jsonl` / `sections.jsonl` to emit JSON events plus unified-text and HTML reports. You can point it at run directories via `python -m zocr.diff --a out/A --b out/B`; see [`zocr/diff/README.md`](zocr/diff/README.md) for layout and CLI details.
- **[FR]** `zocr.diff` compare `cells.jsonl` / `sections.jsonl` afin de produire des événements JSON, un diff texte et un rapport HTML. Il suffit de cibler les dossiers d’exécution avec `python -m zocr.diff --a out/A --b out/B`. L’architecture et l’exemple complet figurent dans [`zocr/diff/README.md`](zocr/diff/README.md).


## クイックスタート / Quickstart / Démarrage rapide
```bash
# 1. インストール / Install / Installation
python -m pip install -e .
# Optional extras:
#   • speed: `python -m pip install -e ".[speed]"`
#   • PDF fallback: `python -m pip install -e ".[pdf]"` (or install Poppler's `pdftoppm`)
#   • Reports: `python -m pip install -e ".[report]"`
#   • API service: `python -m pip install -e ".[api]"`
#   • OCR engines: `python -m pip install -e ".[ocr_tess]"` / `.[ocr_easy]` (optional)

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


## Docker
```bash
# Build (comma-separated extras via build-arg)
docker build -t zocr-suite --build-arg ZOCR_EXTRAS="api" .

# Or bring up the reference API with Docker Compose
export ZOCR_API_KEY="change-me"   # optional (enable auth)
docker compose up --build

# Run the CLI against the current folder (mount as /work)
docker run --rm -v "$PWD:/work" -w /work zocr-suite \
  zocr run --input demo --outdir out_allinone --snapshot --seed 12345
```

## Reference API (FastAPI)
```bash
# Local
python -m pip install -e ".[api]"
zocr-api --host 127.0.0.1 --port 8000
# or: python -m zocr serve --host 127.0.0.1 --port 8000

curl http://127.0.0.1:8000/healthz
curl -F "file=@your.pdf" "http://127.0.0.1:8000/v1/run?domain=invoice"
curl -o zocr_artifacts.zip -F "file=@your.pdf" "http://127.0.0.1:8000/v1/run.zip?domain=invoice"

# Persistent jobs (recommended for internal service use)
curl -F "file=@your.pdf" "http://127.0.0.1:8000/v1/jobs?domain=invoice"
curl "http://127.0.0.1:8000/v1/jobs/<job_id>"
curl -o zocr_job.zip "http://127.0.0.1:8000/v1/jobs/<job_id>/artifacts.zip"
curl -o pipeline_summary.json "http://127.0.0.1:8000/v1/jobs/<job_id>/artifacts/pipeline_summary"

# Optional hardening (env)
export ZOCR_API_KEY="change-me"             # require X-API-Key / Authorization: Bearer
export ZOCR_API_MAX_UPLOAD_MB=50            # upload limit
export ZOCR_API_CONCURRENCY=1               # worker slots
export ZOCR_API_RUN_TIMEOUT_SEC=900         # 0 disables
export ZOCR_API_STORAGE_DIR=/data           # job persistence root (Docker: mount a volume)

curl -H "X-API-Key: $ZOCR_API_KEY" -F "file=@your.pdf" \
  "http://127.0.0.1:8000/v1/run?domain=invoice"
```

## Validation / 検証
```bash
# Validate an outdir (creates zocr.manifest.json if missing)
python -m zocr validate out_allinone --write-manifest
```

## Bench / ベンチ
```bash
python -m zocr bench toy --iterations 5
```

## 統一 CLI / Unified CLI / Interface unifiée
- **[JA]** `python -m zocr run ...` でオーケストレータを実行し、`consensus` や `core` サブコマンドで個別モジュールも呼び出せます。
- **[EN]** `python -m zocr run ...` triggers the orchestrator, while `consensus` and `core` expose the specialised CLIs.
- **[FR]** `python -m zocr run ...` lance l'orchestrateur ; `consensus` et `core` donnent accès aux CLI spécialisées。

| Command | 説明 / Description / Description |
|---------|----------------------------------|
| `python -m zocr run …` | **JA:** パイプライン実行（デフォルト）。<br>**EN:** Run the end-to-end pipeline (default).<br>**FR:** Lance la chaîne complète (par défaut). |
| `python -m zocr pipeline …` | **JA:** `run` と同義、旧来オプション保持。<br>**EN:** Alias of `run`, keeps legacy flags.<br>**FR:** Alias de `run`, conserve les options historiques. |
| `python -m zocr consensus …` | **JA:** OCR/テーブル復元 CLI（デモ・エクスポート）。<br>**EN:** Consensus/table reconstruction CLI.<br>**FR:** CLI pour la reconstruction de tableaux. |
| `python -m zocr core …` | **JA:** マルチモーダルコア（augment/index/query/sql/monitor）。<br>**EN:** Multi-domain core (augment/index/query/sql/monitor).<br>**FR:** Noyau multi-domaine (augment/index/query/sql/monitor). |

## CLI フラグ / CLI Flags / Options CLI
| Flag | 説明 / Description / Description |
|------|----------------------------------|
| `--input` | **JA:** 入力画像/PDF のパス。`demo` で `samples/demo_inputs/` 配下の実ファイルを一括処理。<br>**EN:** Paths to images/PDFs; pass `demo` to sweep every real sample under `samples/demo_inputs/`. <br>**FR:** Chemins vers images/PDF ; `demo` analyse tous les fichiers de `samples/demo_inputs/`. |
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
| `--toy-lite` | **[JA]** Toy OCR を軽量化（demo 入力では自動ON）。<br>**[EN]** Clamp toy OCR sweeps + force numeric columns (auto-enabled for `demo`).<br>**[FR]** Allège le Toy OCR (balayages bornés + colonnes numériques forcées, activé automatiquement pour `demo`). |
| `--toy-sweeps` | **[JA]** Toy OCR の閾値スイープ上限を明示指定。<br>**[EN]** Explicit upper bound for toy OCR threshold sweeps.<br>**[FR]** Borne supérieure explicite pour les balayages de seuil du Toy OCR. |
| `--autocalib [N]` | **[JA]** 表検出の自動キャリブレーションを N ページで実行（値省略時は 3、`0`/未指定で無効）。<br>**[EN]** Auto-calibrate the table detector with N sample pages (defaults to 3 when the flag has no value; pass `0`/omit to skip).<br>**[FR]** Calibre automatiquement le détecteur de tableaux sur N pages (3 si aucune valeur n’est donnée ; `0` ou absence de l’option pour ignorer). |
| `--autotune [N]` | **[JA]** 表検出パラメータを N 試行で自動チューニング（値省略時は 6、`0`/未指定で無効）。<br>**[EN]** Run the unsupervised table autotuner for N trials (defaults to 6 without an explicit value; pass `0`/omit to disable).<br>**[FR]** Lance l’autotune non supervisé sur N essais (6 par défaut sans valeur ; `0` ou absence de l’option pour le désactiver). |
| `--force-numeric-by-header` | **[JA]** ヘッダ名に応じて数量/単価/金額/税率を数値に正規化。<br>**[EN]** Normalize qty/unit price/amount/tax columns according to headers.<br>**[FR]** Normalise les colonnes quantitatives selon les en-têtes. |
| `--ingest-signature` | **[JA]** 別環境での再現ログ（signature JSON）を読み込み差分チェック。<br>**[EN]** Ingest reproducibility signature JSON from another run to compare diffs.<br>**[FR]** Ingère une signature de reproductibilité externe pour comparer les écarts. |
| `--advisor-response` | **[JA]** 外部アドバイザ（LLM等）の助言ファイルを与えて再解析/監視の再実行に接続。<br>**[EN]** Feed advisor (LLM) responses so the orchestrator can trigger reruns based on the advice.<br>**[FR]** Fournit une réponse d’advisor afin de relancer réanalyse/monitoring selon les recommandations. |
| `--tess-unicharset` | **[JA]** Toy OCR の文字集合のみ差し替え。辞書・バイグラムは常に `zocr.resources.domain_dictionary` を使用。<br>**[EN]** Override the toy OCR glyph set; the lexicon/bigram tables always come from the bundled `zocr.resources.domain_dictionary`. <br>**[FR]** Remplace uniquement l'ensemble de glyphes du Toy OCR ; le dictionnaire et les bigrammes proviennent de `zocr.resources.domain_dictionary`. |

> **[JA]** オーケストレータ（`python -m zocr run/pipeline …`）と `python -m zocr consensus …` は `--autocalib [N]` / `--autotune [N]` を共有します。フラグ単体で既定回数（それぞれ 3 / 6）を実行し、`0` または未指定で無効化できます。<br>**[EN]** The orchestrator (`python -m zocr run` / `pipeline`) and `python -m zocr consensus …` share `--autocalib [N]` and `--autotune [N]`: using the flag without a value runs the default counts (3 / 6); passing `0` or omitting the flag disables the pass.<br>**[FR]** L’orchestrateur (`python -m zocr run` / `pipeline`) et `python -m zocr consensus …` partagent `--autocalib [N]` / `--autotune [N]` : sans valeur, les passes par défaut (3 / 6) s’exécutent ; `0` ou l’absence de l’option les désactive.

## サブコマンド / Subcommands / Sous-commandes
- `history --outdir out_invoice --limit 10` — 直近の処理履歴を表示 / show recent history / affiche l'historique récent。
- `summary --outdir out_invoice --keys sql_csv rag_manifest` — 生成物を JSON 出力 / print artifacts / affiche les artefacts。
- `plugins [--stage post_rag]` — 登録済みプラグインを列挙 / list registered hooks / lister les hooks enregistrés。
- `report --outdir out_invoice --open` — 三言語 HTML ダッシュボード生成 / build trilingual HTML dashboard / générer un tableau de bord HTML trilingue。
- `diagnose [--json]` — 依存関係の自己診断（Poppler/Numba/C拡張など）/ dependency self-check for Poppler/Numba/C helpers / autodiagnostic des dépendances (Poppler/Numba/extensions C).

## 仕組み / Mechanics / Fonctionnement
1. **OCR & Consensus** — `zocr.consensus.zocr_consensus` がレイアウト解析とセル信頼度計算を実行。
2. **Export JSONL** — RAG に適した JSONL を出力し、`pipeline_history.jsonl` に記録。
3. **Augment & Index** — `zocr.core.zocr_core` が多領域特徴、BM25、SQL スナップショット、RAG バンドルを構築。
4. **Monitor & Tune** — ヒット率、p95 レイテンシ、失敗率を監視し、必要に応じて自動調整後に再監視。
5. **Report & Plugins** — HTML レポート、要約 JSON、RAG マニフェスト、プラグインフック（`post_export`/`post_index`/`post_monitor`/`post_sql`/`post_rag`）を呼び出し。

各段階は `_safe_step` でガードされ、成功・失敗・経過時間を `pipeline_history.jsonl` に追記します。

- **[JA]** Item/Qty/Unit Price/Amount のスキーマ整形は先頭数行をマージした疑似ヘッダも試し、最初に一致した候補や使用した戦略を `schema_alignment.header_sources` / `strategy_breakdown` に記録するため、ヘッダ欠落ページでも自動整列が効きます。
- **[EN]** The schema rectifier now merges the first rows into synthetic headers before falling back to semantic/heuristic detection and logs the winning header source plus strategy breakdown under `schema_alignment`, so headerless Japanese invoices/estimates still align automatically.
- **[FR]** L’alignement Item/Qty/Unit Price/Amount fusionne désormais les premières lignes pour fabriquer des en-têtes candidats, essaie chaque option puis note la source retenue et la stratégie dans `schema_alignment`, ce qui stabilise les tableaux sans en-tête.

- **[JA]** Export 段階では合議フェーズで得た `row_bands` を再利用し、行分割の精度を維持したままセル OCR を行います。
- **[EN]** During export we reuse the consensus `row_bands` so OCR crops stay aligned with the reconstructed rows.
- **[FR]** Pendant l'export, les `row_bands` issus du consensus sont réutilisés afin d'aligner l'OCR sur les lignes reconstruites.
- **[JA]** toy OCR はエッジ輝度を検知して白地黒字/黒地白字を自動判別し、低信頼セルを削減します。
- **[EN]** The toy OCR now inspects edge brightness to auto-detect inverted text (white-on-black) and cuts down low-confidence cells.
- **[FR]** Le toy OCR détecte désormais automatiquement les inversions (texte clair sur fond sombre) via la brillance des arêtes, réduisant les cellules peu fiables.
- **[JA]** `toy_memory.json`（既定保存先。`ZOCR_TOY_MEMORY` で変更可）にグリフアトラスと N-gram を永続化し、Toy OCR が前回学習した文字や語彙を次回以降も参照します。さらに、実行中は低信頼パッチを短期メモリに保持して新しいグリフを学んだ瞬間に再照合し、`ZOCR_GLYPH_CACHE_LIMIT` / `ZOCR_GLYPH_PENDING_LIMIT` でそのキャッシュサイズを調整できます。N-gram は `ZOCR_NGRAM_EMA_ALPHA` に基づく指数移動平均で忘却と学習のバランスを取り、サマリーの `toy_memory` には読込結果・実行前後の差分・保存状態に加えて、ランタイムキャッシュのヒット率や再試行改善数を含む `recognition` 統計が記録されます。さらに `toy_memory_versioned/summary.json` がエポック番号付きの履歴を蓄積し、最新の `stats` ではエポック数・増分平均・停滞エポック数・直近の高サプライズ比率などを即座に把握でき、`tail` で最近のダイジェストを参照可能です。意図推論はこのメモリ差分と認識統計を参照して再解析やヘッダ強調を自己判断し、`intent_simulations` では `ocr_min_conf` と `lambda_shape` を変更した際の予測値を提示します。
- **[EN]** The toy OCR persists its glyph atlas and N-gram priors to `toy_memory.json` (override via `ZOCR_TOY_MEMORY`) so previously seen glyphs and vocabulary stay available. During a run it now keeps low-confidence patches in a short-term store and replays them whenever a new glyph variant is learned; tune the cache and queue sizes via `ZOCR_GLYPH_CACHE_LIMIT` / `ZOCR_GLYPH_PENDING_LIMIT`. The N-gram model applies an exponential moving average controlled by `ZOCR_NGRAM_EMA_ALPHA` to balance retention and forgetting, and the `toy_memory` summary still lists load/save/delta snapshots while surfacing runtime cache hits, replay gains, and surprisal metrics. The `toy_memory_versioned/summary.json` timeline now records per-epoch deltas together with aggregated `stats` (epoch count, growth averages, stagnant streaks, recent surprisal/runtime replay ratios) and a trimmed `tail` digest so operators can review momentum without opening each epoch file. The intent engine inspects those memory changes plus recognition stats to trigger reanalysis or header focus automatically while `intent_simulations` projects the impact of adjusting `ocr_min_conf` and `lambda_shape`.
- **[FR]** Le toy OCR conserve son atlas de glyphes et ses N-grammes dans `toy_memory.json` (modifiable via `ZOCR_TOY_MEMORY`) afin de réutiliser les caractères déjà vus. Pendant l’exécution il stocke les patches peu fiables dans une mémoire court terme et les réévalue dès qu’un nouveau glyphe est appris ; ajustez la taille des caches avec `ZOCR_GLYPH_CACHE_LIMIT` / `ZOCR_GLYPH_PENDING_LIMIT`. Le modèle de N-grammes applique une moyenne mobile exponentielle (`ZOCR_NGRAM_EMA_ALPHA`) pour doser oubli et apprentissage, et la section `toy_memory` du résumé continue de détailler chargement/sauvegarde/deltas ainsi que les statistiques runtime (hits du cache, relectures réussies, surprisal). Le fichier `toy_memory_versioned/summary.json` consigne désormais chaque exécution avec ses deltas tout en ajoutant des `stats` agrégées (nombre d’époques, croissance moyenne, périodes de stagnation, ratios de surprisal/réutilisation récents) et un `tail` synthétique des derniers epochs, ce qui évite d’ouvrir chaque snapshot. Le moteur d’intentions exploite ces signaux et les statistiques de reconnaissance pour déclencher automatiquement réanalyse ou focus en-têtes, et `intent_simulations` propose des scénarios hypothétiques pour `ocr_min_conf` et `lambda_shape`.
- **[JA]** エクスポートは信頼度だけでなく N-gram の驚異度（surprisal）も監視し、閾値 (`ZOCR_SURPRISAL_REVIEW_THRESHOLD`) を超えたセルを高信頼でも学習キューに積み、`*.signals.json` には低信頼件数と高驚異件数の両方を記録します。意図エンジンは `high_surprisal_ratio` も評価し、文脈的に怪しいセルが増えると同一実行内で自動再解析を起動します。
- **[EN]** The exporter now tracks N-gram surprisal alongside confidence and queues even high-confidence cells for learning when they cross the `ZOCR_SURPRISAL_REVIEW_THRESHOLD`; the `*.signals.json` payload reports both low-confidence and high-surprisal counts. The intent engine inspects `high_surprisal_ratio` so a spike in contextually unlikely cells triggers an immediate reanalysis pass within the same run.
- **[FR]** L’export surveille désormais la surprise N-gramme en plus de la confiance et place dans la file d’apprentissage les cellules dépassant `ZOCR_SURPRISAL_REVIEW_THRESHOLD`, même si leur confiance reste élevée ; le fichier `*.signals.json` indique les volumes faibles-confiances et hautement surprenants. Le moteur d’intentions lit `high_surprisal_ratio` et relance automatiquement la réanalyse lorsqu’une hausse de cellules contextuellement improbables est détectée.
- **[JA]** 再解析 (`reanalyze_learning_jsonl`) は Tesseract があれば追加エンジンとして併用し、未導入でも自前の合成フォールバックが閾値スイープ・単語分割・ポスタライズ・高精度シャープ処理まで試し、Tesseract 風の候補を生成します。文字のゆらぎ辞書で `??I` → `771` などの揺れも補正し、サマリーには `external_engines` に加え `fallback_transform_usage` / `fallback_variant_count` でフォールバックの内訳を記録します。
- **[EN]** The reanalysis stage (`reanalyze_learning_jsonl`) still calls into Tesseract when present, but the synthetic fallback now sweeps adaptive thresholds, performs word segmentation, posterizes, and sharpens aggressively to emulate Tesseract-style outputs when the engine is missing. The ambiguity map continues to remap noisy glyphs such as `??I` → `771`, and the summary exposes both `external_engines` counts and the fallback breakdown via `fallback_transform_usage` / `fallback_variant_count`.
- **[FR]** La phase de réanalyse (`reanalyze_learning_jsonl`) invoque Tesseract lorsqu’il est disponible, et sinon le repli synthétique effectue des balayages de seuils, segmente les mots, applique une posterization et un affûtage poussé afin d’approcher les sorties de Tesseract. La carte d’ambiguïtés convertit toujours des bruits tels que `??I` en `771`, et le résumé détaille désormais `external_engines` ainsi que la ventilation du repli via `fallback_transform_usage` / `fallback_variant_count`.
- **[JA]** `learning_hotspots` / `selective_reanalysis_plan` が `*.learning.jsonl` からホットスポット行・列・セルを抽出し、再解析はそれらの `trace_ids` のみに集中します。`reanalyze_learning_jsonl` のサマリーには `focus_plan` / `focus_stats` が追加され、何件を対象にしたかを即確認できます。
- **[EN]** A hotspot analyzer now scans `*.learning.jsonl` to populate `learning_hotspots` plus a `selective_reanalysis_plan`; reanalysis limits itself to the traced rows/columns so huge tables no longer trigger a full sweep. The `reanalyze_learning_jsonl` summary records the applied `focus_plan` and `focus_stats` so you can see how many cells were targeted.
- **[FR]** L’analyseur de hotspots lit `*.learning.jsonl` et produit `learning_hotspots` ainsi qu’un `selective_reanalysis_plan`, de sorte que la réanalyse ne traite que les cellules/ lignes en cause. Le résumé `reanalyze_learning_jsonl` inclut désormais `focus_plan` et `focus_stats` pour indiquer la couverture réelle.
- **[JA]** 再解析結果は自動で `doc.contextual.reanalyzed.jsonl`（または既存ファイルを書き換え）に反映され、`*.signals.json` には `applied_reanalysis` の集計（改善件数・平均Δなど）を追記します。パイプラインサマリーには `reanalysis_applied` として適用結果が積み上がり、後段の `augment/index/monitor` は更新済みテキストをそのまま利用します。
- **[EN]** Reanalysis outputs now feed straight back into `doc.contextual.reanalyzed.jsonl` (or rewrite the original in place), updating the paired `*.signals.json` with an `applied_reanalysis` block that tracks improved counts and average deltas. The pipeline summary records each pass under `reanalysis_applied`, and downstream augment/index/monitor stages consume the refreshed text automatically.
- **[FR]** Les résultats de réanalyse sont désormais réinjectés dans `doc.contextual.reanalyzed.jsonl` (ou dans le fichier d’origine réécrit), tandis que `*.signals.json` reçoit un bloc `applied_reanalysis` détaillant les gains et la moyenne des deltas. Chaque passage est archivé dans le résumé via `reanalysis_applied`, et les phases augment/index/monitor exploitent d’emblée ces textes mis à jour.
- **[JA]** 検索レイヤーは BM25 + キーワード + 画像類似に加え、`filters` に含まれる数値やキーを直接照合するシンボリックスコアを併用し、Trust@K を押し上げます。
- **[EN]** The retrieval layer now blends BM25 + keyword + image similarity with a symbolic scorer that inspects the structured `filters`, improving Trust@K for downstream RAG agents.
- **[FR]** La couche de recherche combine BM25 + mots-clés + similarité d'image avec un scoreur symbolique basé sur `filters`, ce qui renforce le Trust@K pour les agents RAG.

## 自動ドメイン検出 / Automatic Domain Detection / Détection automatique du domaine
- **[JA]** ファイル名（`samples/demo_inputs/invoice_*.pdf` など）からトークンを抽出し、`DOMAIN_KW` / `_DOMAIN_ALIAS` の別名と突き合わせて初期候補を生成します。OCR 後は JSONL 内テキストとフィルターを走査し、キーワード一致度とヒット率から信頼度を算出します。信頼度が 0.25 未満なら既存ヒントを保持し、`pipeline_summary.json` の `domain_autodetect` に推論経路・信頼度・採用ソースを記録します。
- **[EN]** The orchestrator mines folder/file tokens, maps them through `DOMAIN_KW` and `_DOMAIN_ALIAS`, then refines the guess by scanning the exported JSONL. Confidence scores must clear a 0.25 threshold before overriding prior hints; the full trace lives in `pipeline_summary.json` under `domain_autodetect`.
- **[FR]** L'orchestrateur extrait les jetons des chemins, les compare aux alias/domains connus puis affine la sélection avec le JSONL exporté. Si la confiance reste inférieure à 0,25, l'indice utilisateur est conservé. Le parcours complet est archivé dans `pipeline_summary.json` (`domain_autodetect`).

## 生成物 / Outputs / Résultats
- `doc.zocr.json` — OCR & consensus の主 JSON。
- `doc.mm.jsonl` — マルチモーダル JSONL（RAG / BM25 共用）。
- `rag/` — `export_rag_bundle` によるセル/テーブル/Markdown/マニフェスト。
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
- **[FR]** `pipeline_summary.json` inclut désormais `stage_trace` / `stage_stats`, listant la durée, le statut et un aperçu d’output pour chaque `_safe_step` (identique dans le script monolithique).
- `--print-stage-trace` または `ZOCR_STAGE_TRACE_CONSOLE=1` で実行直後にタイミング表を標準出力へ表示できます（遅延ステージや失敗箇所の即時可視化に便利）。
- **[EN]** Use `--print-stage-trace` or set `ZOCR_STAGE_TRACE_CONSOLE=1` to dump the formatted stage timing table to stdout right after a run, making bottlenecks/failures obvious without opening the JSON summary.
- **[FR]** Activez `--print-stage-trace` ou `ZOCR_STAGE_TRACE_CONSOLE=1` pour afficher aussitôt le tableau des temps d’étape sur la console et repérer goulots d’étranglement / échecs sans consulter le JSON.
- `profile_guard` ブロックには、1ランで変更されたプロファイル項目（既定で最大3件）と、ガードによって拒否/調整されたオーバーライドが記録されます。`ZOCR_PROFILE_MAX_CHANGES` 環境変数で許可数を調整できます。
- `safety_flags.gate_fail_streak` は連続ゲート失敗数を数え、閾値 (`ZOCR_GATE_FAIL_ESCALATE`, 既定3) に到達すると `escalate_to_human` を推奨します。値は `auto_profile.json` にも保存され、次回の run で継続されます。
- **[JA]** Intent の上位には `meta_intent` 層を追加し、「なぜその意図を採択したのか」「どのホットスポットを狙うのか」を `story` と `focus_plan` に記録します。`pipeline_summary.json` や `rag/feedback_request.*` から理由付きをそのまま参照できます。
- **[EN]** A meta-intent layer now narrates why an action was chosen and which hotspots it targets; `pipeline_summary.json` and the generated `rag/feedback_request.*` expose the `meta_intent` story plus its `focus_plan`, giving downstream agents a rationale to follow.
- **[FR]** Une couche méta-intent raconte désormais pourquoi l’action a été retenue et quels hotspots sont visés ; `pipeline_summary.json` ainsi que `rag/feedback_request.*` publient cette `meta_intent` (story + `focus_plan`) pour guider les agents en aval.
- **[JA]** `learning_hotspots` で抽出したセルを `rag/hotspots/*.png` に自動切り出し、`hotspot_gallery` としてサマリーと RAG リクエストに添付します。各エントリには役割（ヘッダ/本文/フッタ）、理由のランク、再解析前後のテキストも記録されるため、JSON を開かずに違和感のあるセルを即確認できます。`ZOCR_HOTSPOT_GALLERY_LIMIT`（既定 12）で枚数を調整できます。
- **[EN]** The orchestrator now crops `learning_hotspots` into `rag/hotspots/*.png`; every gallery entry carries its inferred role (header/body/footer), the ranked reason, and before/after text so reviewers can see the drift without opening JSON. The `hotspot_gallery` block is stored in `pipeline_summary.json` and the RAG request. Tune the export count via `ZOCR_HOTSPOT_GALLERY_LIMIT` (default 12).
- **[FR]** Les hotspots sont découpés en `rag/hotspots/*.png` et chaque entrée mentionne désormais son rôle (en-tête/corps/pied), le rang du signal et les textes avant/après, ce qui permet de repérer les anomalies sans JSON. Le bloc `hotspot_gallery` figure dans le résumé et la requête RAG. Ajustez le nombre d’images via `ZOCR_HOTSPOT_GALLERY_LIMIT` (12 par défaut).
- **[JA]** ギャラリーには `rag/hotspots/gallery.md` も追加され、各セルの位置・テキスト・理由とともに切り出し画像を Markdown で一覧できます。
- **[EN]** A Markdown companion (`rag/hotspots/gallery.md`) now accompanies the PNG crops so advisors can skim every hotspot with its location, observed text, and reasons without opening the JSON.
- **[FR]** Un fichier Markdown (`rag/hotspots/gallery.md`) accompagne la galerie PNG : chaque hotspot y est décrit (position, texte observé, raisons) afin de guider rapidement les conseillers.
- インボイス系ドメインは金額 (`hit_amount>=0.8`) と日付 (`hit_date>=0.5`) の双方が揃わない限り PASS しません。欠損時はゲートが FAIL となり、`gate_reason` で要因を特定できます。
- **[JA]** `monitor.csv` には `trust_amount` / `trust_date` / `trust_mean` を追加し、Top-K に混入した非出典セルの比率を観測できます。`tax_coverage` / `corporate_coverage` でレートが 0 の理由（候補なしなのか失敗か）も判別できます。
- **[EN]** `monitor.csv` now records `trust_amount`, `trust_date`, and `trust_mean`, exposing how many Top-K hits carry proper provenance. Coverage counters (`tax_coverage`, `corporate_coverage`) clarify when rates are zero because no candidates were found.
- **[FR]** `monitor.csv` consigne désormais `trust_amount`, `trust_date` et `trust_mean`, ce qui mesure la part des résultats Top-K dotés de provenance. Les compteurs `tax_coverage` / `corporate_coverage` indiquent si les taux à zéro proviennent d'un manque de candidats.
- Intent 指向のフィードバックでは `intent.action="reanalyze_cells"` が検知されると同一実行内で再解析フローを即時発火し、結果は `intent_runs` と `learning_reanalyzed_jsonl` に反映されます。
- When the intent engine requests `reanalyze_cells`, the orchestrator now fires a same-run reanalysis pass and records the action under `intent_runs` together with the refreshed `learning_reanalyzed_jsonl` path.
- Lorsqu’un intent `reanalyze_cells` est produit, l’orchestrateur déclenche immédiatement la réanalyse et consigne l’action dans `intent_runs` ainsi que le nouveau `learning_reanalyzed_jsonl`.
- Intentシグナルには `intent_simulations` が追加され、Toyメモリの差分から推定した再解析効果や `ocr_min_conf` / `lambda_shape` を上下させた場合の低信頼率・p95予測を提示します。
- The intent payload now includes `intent_simulations`, providing what-if predictions for lowering/raising `ocr_min_conf` and `lambda_shape` plus the expected impact of reanalysis derived from toy-memory deltas.
- Le résumé d’intent expose désormais `intent_simulations`, des scénarios « et si » pour modifier `ocr_min_conf` / `lambda_shape` et l’effet anticipé d’une réanalyse calculé à partir des deltas de mémoire du toy OCR.
- autotune / `learn_from_monitor` が更新した `w_kw` / `w_img` / `ocr_min_conf` / `lambda_shape` を拾い、ヘッダ補完や再走査の微調整ヒントを返します。

## Toy OCR ランタイムノブ / Toy OCR runtime knobs / Commandes Toy OCR
- **[JA]** `ZOCR_TOY_SWEEPS`（既定: 5、`--toy-lite` や `--input demo` では 2〜4）で閾値スイープ回数を固定し、`ZOCR_FORCE_NUMERIC=0` でヘッダ由来の数値強制を無効化できます。
- **[EN]** Bound threshold sweeps via `ZOCR_TOY_SWEEPS` (default 5, auto-clamped to ~2–4 in toy-lite/demo runs) and opt out of header-driven numeric coercion with `ZOCR_FORCE_NUMERIC=0`.
- **[FR]** `ZOCR_TOY_SWEEPS` (par défaut 5, ~2–4 en mode toy-lite/demo) fixe le nombre de balayages ; `ZOCR_FORCE_NUMERIC=0` désactive la coercition numérique basée sur les en-têtes.
- `ZOCR_TOY_MEMORY` で Toy OCR のメモリ保存先を固定でき、`ZOCR_GLYPH_CACHE_LIMIT` / `ZOCR_GLYPH_PENDING_LIMIT` / `ZOCR_NGRAM_EMA_ALPHA` がキャッシュ容量や忘却率を制御します。
- **[JA]** `ZOCR_TEMPLATE_FONT`（任意）でテンプレート照合に使うフォントを指定できます。さらに `ZOCR_GLYPH_FONT`（任意）で Toy OCR のグリフアトラス生成に使うフォントを明示できます（未指定時は `ZOCR_TEMPLATE_FONT` を流用）。日本語トークンや全角記号を Toy OCR で扱いたい場合は、CJK を収録した TrueType/OpenType フォントを指定してください。
- **[EN]** Set `ZOCR_TEMPLATE_FONT` (optional) to pick the font used by token templates. Use `ZOCR_GLYPH_FONT` (optional) to override the font used to build the toy OCR glyph atlas (defaults to `ZOCR_TEMPLATE_FONT`). If you want Toy OCR to handle Japanese tokens/full-width symbols, point these to a CJK-capable TTF/OTF/TTC.
- **[JA]** テンプレート照合の誤爆を抑える場合は `ZOCR_TEMPLATE_MATCH_MIN_SCORE` / `ZOCR_TEMPLATE_MATCH_MIN_MARGIN` を調整できます。テンプレート上書きは `ZOCR_TEMPLATE_OVERRIDE_MIN_CONF` / `ZOCR_TEMPLATE_OVERRIDE_MAX_FINAL_CONF`、テンプレート学習（観測）は `ZOCR_TEMPLATE_OBSERVE_MIN_CONF` / `ZOCR_TEMPLATE_OBSERVE_MIN_QUALITY` / `ZOCR_TEMPLATE_OBSERVE_MAX_LEN` で低信頼の自己強化を抑止できます。
- **[EN]** To reduce template false positives you can tune `ZOCR_TEMPLATE_MATCH_MIN_SCORE` / `ZOCR_TEMPLATE_MATCH_MIN_MARGIN`. Override behaviour is gated by `ZOCR_TEMPLATE_OVERRIDE_MIN_CONF` / `ZOCR_TEMPLATE_OVERRIDE_MAX_FINAL_CONF`, and template observation (self-learning) can be clamped via `ZOCR_TEMPLATE_OBSERVE_MIN_CONF` / `ZOCR_TEMPLATE_OBSERVE_MIN_QUALITY` / `ZOCR_TEMPLATE_OBSERVE_MAX_LEN`.
- `ZOCR_TESS_UNICHARSET` を指定するとグリフ集合のみを差し替えられます。辞書と n-gram は `zocr.resources.domain_dictionary` がまとめたサンプル対応キーワードで常時バンドルされるため、旧来の `--tess-wordlist` / `--tess-bigram-json` は廃止しました。
- **[EN]** Use `ZOCR_TESS_UNICHARSET` (or `--tess-unicharset`) if you need custom glyph coverage; the lexicon/bigram data now ships inside the repo via `zocr.resources.domain_dictionary`, so the legacy `--tess-wordlist` / `--tess-bigram-json` switches are gone.
- **[FR]** `ZOCR_TESS_UNICHARSET` (ou `--tess-unicharset`) reste disponible pour remplacer l’inventaire de glyphes ; le dictionnaire et les bigrammes proviennent dorénavant de `zocr.resources.domain_dictionary`, rendant caducs les anciens commutateurs `--tess-wordlist` / `--tess-bigram-json`.
- **[JA]** 同じ `zocr.resources.domain_dictionary` を consensus exporter や `zocr.core` のキーワードブーストも共有するため、外部辞書や bigram JSON を取り込む想定はスタック全体で廃止しました。
- **[EN]** The same bundled dictionary powers the toy OCR, consensus exporter, and the `zocr.core` retrieval boosts, so no part of the stack expects external wordlists/bigram JSON anymore.
- **[FR]** Ce dictionnaire embarqué alimente aussi bien le Toy OCR que l’exporteur consensus et le noyau `zocr.core`, supprimant toute dépendance aux dictionnaires/bigrammes externes.
- `ZOCR_TESS_DOMAIN` または CLI の `--domain` / パイプラインの domain 設定を指定すると、Toy OCR の内蔵辞書が該当ドメインのキーワード集合に切り替わります。プロファイルや自動判別で domain が確定するとパイプライン側で `ZOCR_TESS_DOMAIN` も自動更新されます。

## PDF レンダリング最適化 / PDF rasterization knobs / Optimisations PDF
- **[JA]** Poppler (`pdftoppm`) が見つからない場合でも、`pypdfium2` がインストールされていれば自動でフォールバックして PDF を PNG に変換します。両方揃っている環境では Poppler が優先されますが、失敗時は即座に pdfium へ切り替わります。
- **[EN]** When Poppler (`pdftoppm`) is missing, the pipeline now falls back to `pypdfium2` automatically, so PDFs can be rasterized without any system packages. If both are available Poppler is used first, with pdfium acting as the safety net.
- **[FR]** Si Poppler (`pdftoppm`) est absent, `pypdfium2` prend automatiquement le relais afin de rasteriser les PDF sans dépendance système. Lorsque les deux sont présents, Poppler reste prioritaire et pdfium sert de filet de sécurité.
- **[JA]** 依存関係ダイアグ (`dependencies.pdf_raster`) で Poppler / pdfium の有効状況とアクティブなバックエンドを表示し、pdfium だけが入っている環境でも「Poppler が無い」という警告が残らないようになりました。あわせて `dependencies.poppler_pdftoppm` / `dependencies.pypdfium2` に個別の可用性とパスが複製されるため、旧来のサマリ閲覧フローでも同じ情報を確認できます。
- **[EN]** The dependency diagnostics (`dependencies.pdf_raster`) now spell out which backend is active (Poppler vs `pypdfium2`), so the summary/logs stop nagging about missing Poppler once the pdfium fallback is installed. The per-backend blocks are also mirrored to `dependencies.poppler_pdftoppm` / `dependencies.pypdfium2`, keeping legacy dashboards and quick JSON viewers in sync.
- **[FR]** Les diagnostics de dépendances (`dependencies.pdf_raster`) précisent désormais quel backend est actif (Poppler ou `pypdfium2`), évitant les alertes « missing » lorsque seul le fallback pdfium est présent. Les blocs par backend sont également recopiés vers `dependencies.poppler_pdftoppm` / `dependencies.pypdfium2`, ce qui aligne les anciens tableaux de bord et les visualisations JSON simplifiées.
- **[JA]** 6 ページ以上の PDF では pdfium 側がデフォルトで並列レンダリング（最大 4 ワーカー、CPU 数に応じて自動調整）を行い、ページ枚数に比例して高速化します。
- **[EN]** For PDFs with ≥6 pages the pdfium path renders pages in parallel (up to four workers by default, auto-tuned to your CPU) which dramatically shortens the raster stage.
- **[FR]** Pour les PDF de 6 pages ou plus, la voie pdfium effectue le rendu en parallèle (jusqu’à quatre workers selon le CPU), accélérant nettement l’étape de rasterisation.
- `ZOCR_PDF_WORKERS` を設定するとワーカー数を固定できます（例: `ZOCR_PDF_WORKERS=2 python -m zocr run ...`）。`ZOCR_PDF_PARALLEL_MIN_PAGES` で並列化を開始する閾値も調整可能です。
- Set `ZOCR_PDF_WORKERS` to clamp the worker count (e.g. `ZOCR_PDF_WORKERS=2 python -m zocr run ...`). Use `ZOCR_PDF_PARALLEL_MIN_PAGES` to raise/lower the page-count threshold.
- Fixez `ZOCR_PDF_WORKERS` pour imposer un nombre précis de workers (ex. `ZOCR_PDF_WORKERS=2 python -m zocr run ...`). Le seuil d’activation peut être ajusté via `ZOCR_PDF_PARALLEL_MIN_PAGES`.
- **[JA]** 長大・ビジュアル密度の高い PDF ではページサイズとページ数から推定したピクセル総量に応じて DPI を自動で下げ、PNG 量産を避けつつ表の精度を維持します。
- **[EN]** For extremely long or highly visual PDFs the rasterizer now lowers the DPI when the projected pixel budget would explode, cutting down the PNG explosion without sacrificing table fidelity.
- **[FR]** Sur les PDF volumineux ou riches en visuels, la rasterisation réduit automatiquement le DPI si le budget de pixels estimé devient excessif, limitant le nombre de PNG tout en conservant la lisibilité des tableaux.
- `ZOCR_PDF_PIXEL_BUDGET` でピクセル上限（既定 3.2e8）を、`ZOCR_PDF_MIN_DPI` で自動縮小時の下限 DPI（既定 120）を変更できます。`ZOCR_PDF_MAX_PAGES` を設定するとレンダリングするページ数そのものを頭打ちでき、`ZOCR_PDF_INSPECT_PAGES` はページサイズ見積りに使うサンプル枚数を制御します。
- Tune the limits via `ZOCR_PDF_PIXEL_BUDGET` (default 3.2e8 pixels) and `ZOCR_PDF_MIN_DPI` (default 120 DPI when throttling kicks in). Set `ZOCR_PDF_MAX_PAGES` to hard-cap the number of rendered pages, and adjust the sampling window with `ZOCR_PDF_INSPECT_PAGES` if you want to inspect more/less pages before estimating sizes.
- Ajustez `ZOCR_PDF_PIXEL_BUDGET` (3,2e8 pixels par défaut) et `ZOCR_PDF_MIN_DPI` (120 DPI mini lorsque la réduction s’active). `ZOCR_PDF_MAX_PAGES` plafonne le nombre de pages rasterisées et `ZOCR_PDF_INSPECT_PAGES` contrôle combien de pages sont échantillonnées pour estimer les dimensions.
- **[JA]** `ZOCR_PDF_MIN_DPI_FLOOR`（既定 72）でハード下限を決め、ソフト下限 (`ZOCR_PDF_MIN_DPI`) が高すぎてピクセル予算を守れない場合でも強制的に DPI を下げられます。
- **[EN]** Use `ZOCR_PDF_MIN_DPI_FLOOR` (default 72) to define the absolute minimum DPI so the rasterizer can still respect the pixel budget even when the soft minimum is higher.
- **[FR]** `ZOCR_PDF_MIN_DPI_FLOOR` (72 par défaut) fixe le plancher absolu, garantissant le respect du budget de pixels même si la borne souple (`ZOCR_PDF_MIN_DPI`) est trop élevée.
- **[JA]** `--snapshot` を付けると `ZOCR_PIPELINE_SNAPSHOT=1` が自動セットされ、`ZOCR_PDF_SNAPSHOT_DPI_PCT`（既定 80%）、`ZOCR_PDF_SNAPSHOT_PIXEL_BUDGET`（既定 2.2e8）、`ZOCR_PDF_SNAPSHOT_MAX_PAGES`（既定無制限）でスナップショット専用の DPI/ピクセル/ページ制限を細かく制御できます。
- **[EN]** When you run with `--snapshot` the orchestrator now sets `ZOCR_PIPELINE_SNAPSHOT=1`, enabling snapshot-specific knobs: `ZOCR_PDF_SNAPSHOT_DPI_PCT` (default 80 %), `ZOCR_PDF_SNAPSHOT_PIXEL_BUDGET` (default 2.2e8), and `ZOCR_PDF_SNAPSHOT_MAX_PAGES` (disabled by default) to throttle DPI, total pixels, or page count just for traced runs.
- **[FR]** Avec `--snapshot`, l’orchestrateur active `ZOCR_PIPELINE_SNAPSHOT=1`, ce qui permet d’appliquer des limites dédiées (`ZOCR_PDF_SNAPSHOT_DPI_PCT`=80 % par défaut, `ZOCR_PDF_SNAPSHOT_PIXEL_BUDGET`=2,2e8, `ZOCR_PDF_SNAPSHOT_MAX_PAGES` désactivé) pour réduire le DPI, le volume de pixels ou le nombre de pages uniquement durant les captures.
- **[EN]** Set `ZOCR_TESS_DOMAIN` (or pass `--domain` to the consensus CLI / orchestrator) to clamp the bundled lexicon to a specific domain keyword set. The pipeline writes this env var automatically whenever its profile or autodetector selects a domain.
- **[FR]** Utilisez `ZOCR_TESS_DOMAIN` (ou l’option `--domain` côté CLI/pipe) pour restreindre le dictionnaire embarqué au jeu de mots-clés d’un domaine. L’orchestrateur met à jour cette variable dès qu’un domaine est choisi par le profil ou la détection automatique.
- **[JA]** 追加指定なしでもリポジトリ同梱の tesslite セット（JP/EN インボイス語彙）が自動で読み込まれます。`--tess-*` / `ZOCR_TESS_*` を指定すると上書きされ、`ZOCR_TESSLITE_DISABLE_BUILTIN=1` で無効化できます。<br>**[EN]** A bundled tesslite glyph/dictionary set now loads automatically—override it with `--tess-*` / `ZOCR_TESS_*` or disable via `ZOCR_TESSLITE_DISABLE_BUILTIN=1`. <br>**[FR]** Un jeu tesslite intégré est actif par défaut ; remplacez-le via `--tess-*` / `ZOCR_TESS_*` ou désactivez-le avec `ZOCR_TESSLITE_DISABLE_BUILTIN=1`.
- **[JA]** motion prior / tesslite / lexical & numeric confidence boost / N-gram EMA / hotspot 検出 / view 生成 / intent simulations はすべて既定でオンになり、`pipeline_summary.json` / `toy_feature_defaults` に適用状況が記録されます。追加の環境変数なしで Toy エンジンのフル機能が動作し、必要に応じてサマリで確認可能です。<br>**[EN]** Motion priors, tesslite dictionaries, lexical & numeric confidence boosts, the N-gram EMA, hotspot detection, microscope/X-ray view generation, and intent simulations are now enabled by default—no extra env vars needed—and the `toy_feature_defaults` block in `pipeline_summary.json` records which knobs were applied. <br>**[FR]** Les motion priors, dictionnaires tesslite, boosts lexical/numérique, EMA des N-grammes, détection de hotspots, vues microscope/X-ray et simulations d’intent sont tous actifs par défaut sans variables d’environnement supplémentaires ; le bloc `toy_feature_defaults` du `pipeline_summary.json` consigne l’état de chaque fonction.
- `--toy-lite` または demo 入力では数値列の強制正規化と sweep クランプが既定で有効になり、`pipeline_summary.json` の `toy_runtime_config` と `last_export_stats` に適用結果が保存されます。
- `--autocalib` / `--autotune` フラグを有効にすると、事前の表キャリブレーション / オートチューニング結果が `pipeline_summary.json` の `table_autocalib` / `table_autotune` として記録され、採用されたパラメータは `table_params` に保存されます。<br>**[EN]** When the optional `--autocalib` / `--autotune` flags are used, the pre-export calibration/tuning summaries are written to `pipeline_summary.json` (`table_autocalib` / `table_autotune`) and the applied values land under `table_params`.<br>**[FR]** Lorsque vous activez `--autocalib` / `--autotune`, les résultats de calibration/auto-réglage sont consignés dans `pipeline_summary.json` (`table_autocalib` / `table_autotune`) et les paramètres retenus figurent dans `table_params`.

## Export 進捗と高速化 / Export progress & acceleration / Export : progression et accélérations
- `ZOCR_EXPORT_OCR` で Export 内の OCR バックエンドを切り替えられます（例: `fast` でセル OCR をスキップし構造のみ書き出し、`toy` / `tesseract` で再解析）。
- **[JA]** `zocr consensus export` では画像フォルダ／ワイルドカード／PDF パスをそのまま渡しても自動でビットマップを列挙し、PDF の場合は Poppler/pdfium でその場レンダリングします。パスが壊れてページ画像を開けなかった場合は `missing page bitmaps` 警告に候補とページ番号をまとめて表示するため、`doc.contextual.jsonl` が空のままでも原因を即座に特定できます。<br>**[EN]** The standalone `zocr consensus export` command now walks directories, glob patterns, or even raw PDF paths for you—PDFs are rasterised on demand via Poppler/pdfium—and it logs `missing page bitmaps` with the attempted candidates so broken paths can be fixed before the JSONL ends up empty.<br>**[FR]** La commande autonome `zocr consensus export` accepte désormais dossiers, motifs glob ou PDF bruts : les PDF sont rasterisés à la volée (Poppler/pdfium) et chaque page introuvable est signalée (`missing page bitmaps`) avec les chemins testés pour éviter des JSONL vides sans diagnostic.
- **[JA]** セル輸出用のガード (`ZOCR_EXPORT_GUARD_MS` / `--export-guard-ms`) は既定で無効になりました。値を指定した場合も処理済みセルは `guard_timeout` フラグ付きで JSONL に残るため、長大テーブルでも途中結果が失われません。
- **[EN]** The per-table export guard (`ZOCR_EXPORT_GUARD_MS` / `--export-guard-ms`) is now opt-in. Leave it unset to let slow tables finish; if you do set a limit the exporter still writes whatever cells were processed and tags them with `guard_timeout` for downstream review.
- **[FR]** La garde d’export par table (`ZOCR_EXPORT_GUARD_MS` / `--export-guard-ms`) est désactivée par défaut. Fixez-la uniquement si vous souhaitez un plafond temporel ; même en cas de dépassement, les cellules déjà traitées sont écrites avec l’étiquette `guard_timeout` pour faciliter les revues.
- `ZOCR_ALLOW_PYTESSERACT=0` もしくは `--no-allow-pytesseract` を指定すると外部 pytesseract 呼び出しを完全停止できます（既定は許可で、`--allow-pytesseract` / `ZOCR_ALLOW_PYTESSERACT=1` は明示的な上書きとして残しています）。
- Set `ZOCR_ALLOW_PYTESSERACT=0` or pass `--no-allow-pytesseract` to fully disable pytesseract; it is now allowed by default, and `--allow-pytesseract` / `ZOCR_ALLOW_PYTESSERACT=1` remain as explicit opt-ins if you need to override other settings.
- Export 時の行バンド再シード（motion prior）は既定で常時有効になりました。`--no-motion-prior` や `ZOCR_EXPORT_MOTION_PRIOR=0` で無効化できます。<br>**[EN]** Motion-prior reseeding between toy export sweeps is now enabled by default; opt out via `--no-motion-prior` or `ZOCR_EXPORT_MOTION_PRIOR=0`. <br>**[FR]** Le motion prior est désormais actif par défaut ; utilisez `--no-motion-prior` ou `ZOCR_EXPORT_MOTION_PRIOR=0` pour revenir à l’exploration exhaustive.
- `ZOCR_EXPORT_PROGRESS=1` と `ZOCR_EXPORT_LOG_EVERY=100`（任意）でセル処理数の進捗ログを標準出力に流し、長大なグリッドでも固まって見えません。
- `ZOCR_EXPORT_MAX_CELLS` を指定すると巨大テーブルをサンプリングできます。進捗ログ有効時は `last_export_stats()` / `pipeline_summary.json` にページ数・セル数・数値強制件数・処理秒数が残ります。
- Toy OCR と組み合わせる場合は `ZOCR_TOY_SWEEPS=2 ZOCR_EXPORT_OCR=fast` で 4 ページ超のインボイスでも即時に JSONL/SQL/RAG を生成できます。

## アドバイザ/再現性ループ / Advisor & reproducibility loops / Boucles advisor + reproductibilité
- **[JA]** `--ingest-signature foreign_summary.json` で別環境の `pipeline_summary.json` を読み込み、差分を `pipeline_summary.json` の `reproducibility` ブロックに記録します。
- **[EN]** Supply `--advisor-response advisor.jsonl` (JSON or plaintext) to ingest LLM/agent feedback; parsed actions such as `reanalyze_cells` or `rerun_monitor` feed directly into the rerun controller and their effects are logged under `advisor_ingest` / `advisor_applied`.
- **[FR]** `pipeline_summary.json` expose désormais `intent_stories`, `toy_learning_story`, `advisor_ingest`, `advisor_applied`, `reanalysis_outcome`, etc., ce qui documente pourquoi des réanalyses/monitors ont été relancés.
- 再現署名は `pipeline_summary.json` の `repro_signature` に保存され、外部共有→`--ingest-signature` で読み込むことで「期待値との差分」を自動で可視化できます。

## 対応ドメイン / Supported Domains / Domaines pris en charge
- インボイス (JP/EN/FR)、見積書、納品書、領収書、契約書、購入注文書、経費精算、タイムシート、出荷案内、銀行明細、公共料金請求書。
- 医療領収書に加え **医療請求 (medical_bill)**、**通関申告 (customs_declaration)**、**助成金申請 (grant_application)**、**搭乗券 (boarding_pass)**、**銀行明細 (bank_statement)**、**公共料金 (utility_bill)**、**保険金請求 (insurance_claim)**、**税務申告 (tax_form)**、**給与明細 (payslip)**、**出荷案内 (shipping_notice)**、**経費精算 (expense_report)** を新たに追加。既存の **賃貸借契約**、**ローン明細**、**旅行行程** も強化済みです。
- 各ドメインは `DOMAIN_KW`/`DOMAIN_DEFAULTS`/`DOMAIN_SUGGESTED_QUERIES` を共有し、RAG 推奨クエリやウォームアップ検索を自動設定します。
- **[JA]** インボイス系の既定 `ocr_min_conf` を 0.55 前後に調整し、薄い印字や淡色背景でもセルを保持しやすくしています。
- **[EN]** Invoice defaults now use an `ocr_min_conf` around 0.55 so faint print and pastel backgrounds stay captured.
- **[FR]** Les profils facture adoptent désormais un `ocr_min_conf` proche de 0,55 afin de conserver les textes pâles ou sur fonds pastels.

## RAG 連携 / RAG Integration / Intégration RAG
- `export_rag_bundle` が Markdown ダイジェスト、セル JSONL、テーブル別セクション、推奨クエリを `rag/manifest.json` にまとめます。
- サマリー (`summary` サブコマンド) から `rag_markdown` や `rag_suggested_queries` を取得し、下流エージェントに渡せます。
- `post_rag` フックでカスタム転送やストレージ連携を差し込めます。
- **[JA]** 各セルは `trace`（`doc=...;page=...;row=...`）と `<fact trace="...">text</fact>` を保持し、`rag_trace_schema` / `rag_fact_tag_example` で下流 LLM へ「出典必須」のプロンプトを構築できます。
- **[EN]** Each cell now ships with a `trace` string (`doc=...;page=...;row=...`) plus an immutable `<fact trace="...">…</fact>` tag so RAG/LLM stacks can demand provenance; see `rag_trace_schema` and `rag_fact_tag_example` in the summary.
- **[FR]** Chaque cellule fournit désormais une `trace` (`doc=...;page=...;row=...`) et une balise `<fact trace="...">…</fact>` pour imposer la provenance côté LLM ; consultez `rag_trace_schema` et `rag_fact_tag_example` dans le résumé.
- **Feedback loop / フィードバックループ** — `rag/manifest.json` に `"feedback": {"profile_overrides": {...}, "actions": ["reanalyze_cells"], "notes": "..."}` を追記すると、次回 `--resume` 実行時に自動で読み込みます。`auto_profile.json` へ上書きし、`actions` には `reanalyze_cells` / `rerun_monitor` / `rerun_augment` などを記載できます。
- `--rag-feedback path/to/manifest.json` を指定すると、別ディレクトリで生成した RAG マニフェストでも同じ処理（プロファイル上書き + rerun 指示）を適用できます。適用結果は `pipeline_summary.json` の `rag_feedback`, `rag_feedback_actions(_applied)` に記録されます。
- `rag/feedback_request.json` / `.md` が各実行後に生成され、現在のメトリクス・意図・進捗と共に「manifest の feedback ブロックをどのように編集すべきか」を JA/EN のガイド付きで提示します。LLM や外部エージェントに渡せば、そのまま `feedback` を追記して `--resume` で再投入できます。
- 生成されるフィードバックリクエストには `meta_intent`・`learning_hotspots`・`selective_reanalysis_plan` も含まれるため、外部 LLM が「どのセルを直せばよいか」を即座に把握できます。
- The feedback bundle now embeds the latest `meta_intent`, `learning_hotspots`, and `selective_reanalysis_plan`, so external LLMs can reason about root causes and suggest precise fixes instead of scanning the entire run.
- Le paquet de feedback inclut désormais `meta_intent`, `learning_hotspots` et `selective_reanalysis_plan`, ce qui permet à un LLM externe d’identifier immédiatement les zones à corriger.
- **[JA]** `rag/feedback_request.*` にはホットスポットや meta-intent を踏まえた質問テンプレ（低信頼率をどう下げるか、どの trace を直すべきか等）が自動生成され、人間/LLM がそのまま回答ブロックとして追記できます。
- **[EN]** The `rag/feedback_request.*` bundle now ships with reviewer questions tailored to the current hotspots, meta-intent story, and low-confidence metrics, so a human or LLM can answer them directly before pasting the response back into the manifest.
- **[FR]** Le paquet `rag/feedback_request.*` comprend désormais des questions ciblées (hotspots, meta-intent, faible confiance) afin qu’un humain ou un LLM puisse y répondre immédiatement avant de réinjecter le feedback dans le manifeste.
- `hotspot_gallery` (JSON + `rag/hotspots/*.png`) is also exported so advisors can see the offending cells without reopening the PDFs; disable or resize the gallery via `ZOCR_HOTSPOT_GALLERY_LIMIT`.
- **[JA]** `rag/conversation.jsonl` にパイプライン→LLM→RAG 間の対話履歴を追記します。生成した feedback request / 取り込んだ manifest / advisor 回答が順番に記録され、`pipeline_summary.json` の `rag_conversation` から最新エントリを辿れます。
- **[EN]** Every run now appends to `rag/conversation.jsonl`, logging the emitted feedback requests plus any ingested manifest/advisor responses. `pipeline_summary.json` exposes the path + last entry under `rag_conversation`, so external agents can replay or extend the dialog.
- **[FR]** Chaque exécution ajoute désormais les requêtes de feedback et les réponses manifest/advisor ingérées dans `rag/conversation.jsonl`; le résumé (`rag_conversation`) expose le chemin et la dernière entrée afin que les agents externes puissent poursuivre le dialogue.

## ビジュアライゼーション / Visualisation / Visualisation
- 4 パネルのマイクロスコープ（原画、シャープ、二値、勾配）と X-Ray オーバーレイを自動生成。
- `--views-log` で生成履歴を CSV 追記し、監視や QA に利用可能。

## サンプル素材 / Sample Assets / Ressources d'exemple
- `samples/demo_inputs/` にファイルを配置すると、`--input demo` がそれらを取り込みます。
- フォルダは空でも構いません。クローン直後は同梱の合成サンプルが利用されます。
- `samples/README.md` に多言語で手順をまとめています。
- 旧来の `samples/<domain>/` フォルダは廃止し、`samples/demo_inputs/` に一本化しました。各ドメインのキーワードや `--domain` 推奨値は `samples/README.md` 内の早見表に集約しています。

## 依存関係 / Dependencies / Dépendances
- Python 3.10+
- NumPy, Pillow（コア機能）
- Numba（任意、加速）
- Poppler (`pdftoppm`) または `pypdfium2`（PDF 入力時に必要な場合あり）
- pandas（任意、一部の集計/可視化で利用）
- pytesseract / easyocr（任意、Toy OCR の代替エンジン）

## 単一ファイル版（レガシー）/ Legacy monolith / Monolithe historique
- **[JA]** 本 OSS リポジトリはモジュラー版（`zocr/`）を主対象とし、単一ファイル版は同梱していません。
- **[EN]** This OSS repo focuses on the modular `zocr/` package; a monolithic single-file bundle is not shipped here.
- **[FR]** Ce dépôt OSS se concentre sur le paquet modulaire `zocr/` ; le bundle monolithique n’est pas fourni ici.
