# Z-OCR Suite / Z-OCR スイート / Suite Z-OCR

## 概要 / Overview / Aperçu
- **[JA]** `zocr/` パッケージ（`consensus`, `core`, `orchestrator`）を中心に、OCR→拡張→インデックス→監視→調整→レポートまでをワンコマンドで連結します。`zocr_allinone_merged_plus.py` は互換レガシーとして同梱します。
- **[EN]** The repo focuses on the modular `zocr/` package (`consensus`, `core`, `orchestrator`) that chains OCR → augmentation → indexing → monitoring → tuning → reporting. The legacy `zocr_allinone_merged_plus.py` remains as a drop-in backup.
- **[FR]** Le paquet modulaire `zocr/` (`consensus`, `core`, `orchestrator`) relie OCR → augmentation → indexation → surveillance → réglage → rapport via une seule commande. Le fichier unique `zocr_allinone_merged_plus.py` est conservé pour compatibilité.

## レイアウト / Layout / Structure
```
zocr/
  consensus/zocr_consensus.py    # OCR + table reconstruction helpers
  core/zocr_core.py              # augmentation, BM25, monitoring, SQL & RAG export
  orchestrator/zocr_pipeline.py  # CLI pipeline orchestrator + resume/watchdog/reporting
samples/
  demo_inputs/                   # place your PDFs/PNGs here for quick demos
README.md
zocr_allinone_merged_plus.py     # legacy single-file bundle (same features)
```

## クイックスタート / Quickstart / Démarrage rapide
```bash
# 1. 依存関係 / Dependencies / Dépendances
python -m pip install numpy pillow tqdm numba

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
| `--input` | **JA:** 入力画像/PDF のパス。`demo` は合成デモ＋`samples/demo_inputs/`。<br>**EN:** Paths to images/PDFs; `demo` mixes the synthetic example with anything in `samples/demo_inputs/`.<br>**FR:** Chemins vers images/PDF ; `demo` combine l'exemple synthétique et les fichiers dans `samples/demo_inputs/`. |
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

- **[JA]** Export 段階では合議フェーズで得た `row_bands` を再利用し、行分割の精度を維持したままセル OCR を行います。
- **[EN]** During export we reuse the consensus `row_bands` so OCR crops stay aligned with the reconstructed rows.
- **[FR]** Pendant l'export, les `row_bands` issus du consensus sont réutilisés afin d'aligner l'OCR sur les lignes reconstruites.
- **[JA]** toy OCR はエッジ輝度を検知して白地黒字/黒地白字を自動判別し、低信頼セルを削減します。
- **[EN]** The toy OCR now inspects edge brightness to auto-detect inverted text (white-on-black) and cuts down low-confidence cells.
- **[FR]** Le toy OCR détecte désormais automatiquement les inversions (texte clair sur fond sombre) via la brillance des arêtes, réduisant les cellules peu fiables.
- **[JA]** `toy_memory.json`（既定保存先。`ZOCR_TOY_MEMORY` で変更可）にグリフアトラスと N-gram を永続化し、Toy OCR が前回学習した文字や語彙を次回以降も参照します。サマリーの `toy_memory` には読込結果（`load`）と実行前後のスナップショット差分（`before_run` / `after_run` / `delta_run`）、保存結果（`save`）が記録され、Toy の学習量を可視化できます。
- **[EN]** The toy OCR persists its glyph atlas and N-gram priors to `toy_memory.json` (override via `ZOCR_TOY_MEMORY`), so previously seen glyphs/vocabulary are reused. The `toy_memory` block in the summary now captures the load outcome, before/after snapshots with per-run deltas, and the final save status so you can quantify how much the toy learned.
- **[FR]** Le toy OCR conserve son atlas de glyphes et ses N-grammes dans `toy_memory.json` (chemin modifiable via `ZOCR_TOY_MEMORY`), réutilisant les caractères observés. La section `toy_memory` du résumé consigne désormais le résultat du chargement, les instantanés avant/après avec les deltas par exécution, ainsi que l’état de sauvegarde pour mesurer l’apprentissage du toy.
- **[JA]** 再解析 (`reanalyze_learning_jsonl`) は Tesseract があれば追加エンジンとして併用し、未導入でも自前の合成フォールバックが閾値スイープ・単語分割・ポスタライズ・高精度シャープ処理まで試し、Tesseract 風の候補を生成します。文字のゆらぎ辞書で `??I` → `771` などの揺れも補正し、サマリーには `external_engines` に加え `fallback_transform_usage` / `fallback_variant_count` でフォールバックの内訳を記録します。
- **[EN]** The reanalysis stage (`reanalyze_learning_jsonl`) still calls into Tesseract when present, but the synthetic fallback now sweeps adaptive thresholds, performs word segmentation, posterizes, and sharpens aggressively to emulate Tesseract-style outputs when the engine is missing. The ambiguity map continues to remap noisy glyphs such as `??I` → `771`, and the summary exposes both `external_engines` counts and the fallback breakdown via `fallback_transform_usage` / `fallback_variant_count`.
- **[FR]** La phase de réanalyse (`reanalyze_learning_jsonl`) invoque Tesseract lorsqu’il est disponible, et sinon le repli synthétique effectue des balayages de seuils, segmente les mots, applique une posterization et un affûtage poussé afin d’approcher les sorties de Tesseract. La carte d’ambiguïtés convertit toujours des bruits tels que `??I` en `771`, et le résumé détaille désormais `external_engines` ainsi que la ventilation du repli via `fallback_transform_usage` / `fallback_variant_count`.
- **[JA]** 検索レイヤーは BM25 + キーワード + 画像類似に加え、`filters` に含まれる数値やキーを直接照合するシンボリックスコアを併用し、Trust@K を押し上げます。
- **[EN]** The retrieval layer now blends BM25 + keyword + image similarity with a symbolic scorer that inspects the structured `filters`, improving Trust@K for downstream RAG agents.
- **[FR]** La couche de recherche combine BM25 + mots-clés + similarité d'image avec un scoreur symbolique basé sur `filters`, ce qui renforce le Trust@K pour les agents RAG.

## 自動ドメイン検出 / Automatic Domain Detection / Détection automatique du domaine
- **[JA]** ファイル名（`samples/invoice/...` など）からトークンを抽出し、`DOMAIN_KW` / `_DOMAIN_ALIAS` の別名と突き合わせて初期候補を生成します。OCR 後は JSONL 内テキストとフィルターを走査し、キーワード一致度とヒット率から信頼度を算出します。信頼度が 0.25 未満なら既存ヒントを保持し、`pipeline_summary.json` の `domain_autodetect` に推論経路・信頼度・採用ソースを記録します。
- **[EN]** The orchestrator mines folder/file tokens, maps them through `DOMAIN_KW` and `_DOMAIN_ALIAS`, then refines the guess by scanning the exported JSONL. Confidence scores must clear a 0.25 threshold before overriding prior hints; the full trace lives in `pipeline_summary.json` under `domain_autodetect`.
- **[FR]** L'orchestrateur extrait les jetons des chemins, les compare aux alias/domains connus puis affine la sélection avec le JSONL exporté. Si la confiance reste inférieure à 0,25, l'indice utilisateur est conservé. Le parcours complet est archivé dans `pipeline_summary.json` (`domain_autodetect`).

## 生成物 / Outputs / Résultats
- `doc.zocr.json` — OCR & consensus の主 JSON。
- `doc.mm.jsonl` — マルチモーダル JSONL（RAG / BM25 共用）。
- `rag/` — `export_rag_bundle` によるセル/テーブル/Markdown/マニフェスト。
- `sql/` — `sql_export` で生成される CSV とスキーマ（`trace` 列で doc/page/table/row/col を Excel から参照可能）。
- `views/` — マイクロスコープ 4 分割＋X-Ray オーバーレイ。
- `reanalyze/` — 低信頼セルの再解析 JSONL（`*.summary.json` には `external_engines` と `fallback_transform_usage` / `fallback_variant_count` を含む詳細統計） / Reanalysis JSONL for low-confidence cells (the accompanying `*.summary.json` captures `external_engines` plus `fallback_transform_usage` / `fallback_variant_count`) / Ré-analyses JSONL des cellules peu fiables (le `*.summary.json` expose `external_engines` ainsi que `fallback_transform_usage` / `fallback_variant_count`).
- `toy_memory.json` — Toy OCR の記憶スナップショット（グリフアトラスと N-gram）。`ZOCR_TOY_MEMORY` 環境変数で場所を固定可能。
- `toy_memory.json` — Snapshot of the toy OCR memory (glyph atlas + N-grams). Set `ZOCR_TOY_MEMORY` to pin the storage path.
- `toy_memory.json` — Instantané de la mémoire du toy OCR (atlas de glyphes + N-grammes). Utilisez `ZOCR_TOY_MEMORY` pour fixer l’emplacement.
- `pipeline_summary.json` — すべての成果物と依存診断をまとめた要約（`rag_*`, `sql_*`, `views`, `dependencies`, `report_path` など）。
- `rag_trace_schema`, `rag_fact_tag_example` — サマリー内で RAG トレーサの仕様と `<fact ...>` タグ例を公開。
- `monitor.csv` — UTF-8 (BOM 付き) で出力し、Excel/Numbers でも文字化けなく開けます。
- `pipeline_meta.json` — `--snapshot` 有効時の環境情報。
- `pipeline_report.html` — trilingual ダッシュボード（`report` サブコマンドでも再生成可）。

## モニタリング洞察 / Monitoring Insights / Analyse de la surveillance
- `pipeline_summary.json` の `insights` は構造・ゲート・プロファイルの3本立てで、over/under・TEDS・行外れ率や Hit@K を数値付きで提示します。
- インボイス系ドメインは金額 (`hit_amount>=0.8`) と日付 (`hit_date>=0.5`) の双方が揃わない限り PASS しません。欠損時はゲートが FAIL となり、`gate_reason` で要因を特定できます。
- **[JA]** `monitor.csv` には `trust_amount` / `trust_date` / `trust_mean` を追加し、Top-K に混入した非出典セルの比率を観測できます。`tax_coverage` / `corporate_coverage` でレートが 0 の理由（候補なしなのか失敗か）も判別できます。
- **[EN]** `monitor.csv` now records `trust_amount`, `trust_date`, and `trust_mean`, exposing how many Top-K hits carry proper provenance. Coverage counters (`tax_coverage`, `corporate_coverage`) clarify when rates are zero because no candidates were found.
- **[FR]** `monitor.csv` consigne désormais `trust_amount`, `trust_date` et `trust_mean`, ce qui mesure la part des résultats Top-K dotés de provenance. Les compteurs `tax_coverage` / `corporate_coverage` indiquent si les taux à zéro proviennent d'un manque de candidats.
- Intent 指向のフィードバックでは `intent.action="reanalyze_cells"` が検知されると同一実行内で再解析フローを即時発火し、結果は `intent_runs` と `learning_reanalyzed_jsonl` に反映されます。
- When the intent engine requests `reanalyze_cells`, the orchestrator now fires a same-run reanalysis pass and records the action under `intent_runs` together with the refreshed `learning_reanalyzed_jsonl` path.
- Lorsqu’un intent `reanalyze_cells` est produit, l’orchestrateur déclenche immédiatement la réanalyse et consigne l’action dans `intent_runs` ainsi que le nouveau `learning_reanalyzed_jsonl`.
- autotune / `learn_from_monitor` が更新した `w_kw` / `w_img` / `ocr_min_conf` / `lambda_shape` を拾い、ヘッダ補完や再走査の微調整ヒントを返します。

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

## ビジュアライゼーション / Visualisation / Visualisation
- 4 パネルのマイクロスコープ（原画、シャープ、二値、勾配）と X-Ray オーバーレイを自動生成。
- `--views-log` で生成履歴を CSV 追記し、監視や QA に利用可能。

## サンプル素材 / Sample Assets / Ressources d'exemple
- `samples/demo_inputs/` にファイルを配置すると、`--input demo` がそれらを取り込みます。
- フォルダは空でも構いません。クローン直後は同梱の合成サンプルが利用されます。
- `samples/README.md` に多言語で手順をまとめています。
- `samples/invoice/`, `samples/purchase_order/`, `samples/medical_bill/`, `samples/customs_declaration/`, `samples/grant_application/`, `samples/boarding_pass/`, `samples/bank_statement/`, `samples/utility_bill/`, `samples/insurance_claim/`, `samples/tax_form/`, `samples/payslip/`, `samples/shipping_notice/`, `samples/expense_report/` など、それぞれの README が推奨ドメインや注目フィールドを案内します。

## 依存関係 / Dependencies / Dépendances
- Python 3.9+
- NumPy, Pillow, tqdm（コア機能）
- Numba（任意、BM25 DF の並列化）
- Poppler など PDF レンダラ（PDF 入力時に必要な場合あり）

## バックアップの単一ファイル版 / Single-file Backup / Fichier unique de secours
- 既存ワークフローが `zocr_allinone_merged_plus.py` に依存している場合、そのまま利用できます。
- モジュラー版と同じ CLI/サブコマンド/プラグイン/レポート機能を保持しています。
