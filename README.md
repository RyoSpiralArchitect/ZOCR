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
python -m zocr.orchestrator.zocr_pipeline --input demo --domain invoice --snapshot --seed 12345

# 4. 途中再開 / Resume after failure / Reprendre après un échec
python -m zocr.orchestrator.zocr_pipeline --outdir out_invoice --resume --seed 12345
```

> `python -m zocr.orchestrator.zocr_pipeline run ...` も同義です。All commands below accept either form.

## CLI フラグ / CLI Flags / Options CLI
| Flag | 説明 / Description / Description |
|------|----------------------------------|
| `--input` | **JA:** 入力画像/PDF のパス。`demo` は合成デモ＋`samples/demo_inputs/`。<br>**EN:** Paths to images/PDFs; `demo` mixes the synthetic example with anything in `samples/demo_inputs/`.<br>**FR:** Chemins vers images/PDF ; `demo` combine l'exemple synthétique et les fichiers dans `samples/demo_inputs/`. |
| `--outdir` | 出力先 / Output directory / Répertoire de sortie。既定: `out_allinone`. |
| `--dpi` | PDF レンダリング時の DPI / DPI for PDF rendering / DPI pour le rendu PDF. |
| `--domain` | ドメインヒント（`invoice`, `bank_statement`, `tax`, など）/ Domain hint / Indice de domaine. |
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

## 仕組み / Mechanics / Fonctionnement
1. **OCR & Consensus** — `zocr.consensus.zocr_consensus` がレイアウト解析とセル信頼度計算を実行。
2. **Export JSONL** — RAG に適した JSONL を出力し、`pipeline_history.jsonl` に記録。
3. **Augment & Index** — `zocr.core.zocr_core` が多領域特徴、BM25、SQL スナップショット、RAG バンドルを構築。
4. **Monitor & Tune** — ヒット率、p95 レイテンシ、失敗率を監視し、必要に応じて自動調整後に再監視。
5. **Report & Plugins** — HTML レポート、要約 JSON、RAG マニフェスト、プラグインフック（`post_export`/`post_index`/`post_monitor`/`post_sql`/`post_rag`）を呼び出し。

各段階は `_safe_step` でガードされ、成功・失敗・経過時間を `pipeline_history.jsonl` に追記します。

## 生成物 / Outputs / Résultats
- `doc.zocr.json` — OCR & consensus の主 JSON。
- `doc.mm.jsonl` — マルチモーダル JSONL（RAG / BM25 共用）。
- `rag/` — `export_rag_bundle` によるセル/テーブル/Markdown/マニフェスト。
- `sql/` — `sql_export` で生成される CSV とスキーマ。
- `views/` — マイクロスコープ 4 分割＋X-Ray オーバーレイ。
- `pipeline_summary.json` — すべての成果物をまとめた要約（`rag_*`, `sql_*`, `views`, `report_path` など）。
- `pipeline_meta.json` — `--snapshot` 有効時の環境情報。
- `pipeline_report.html` — trilingual ダッシュボード（`report` サブコマンドでも再生成可）。

## 対応ドメイン / Supported Domains / Domaines pris en charge
- インボイス (JP/EN/FR)、見積書、納品書、領収書、契約書、購入注文書。
- 経費精算、タイムシート、出荷案内、医療領収書、銀行明細、公共料金請求書。
- 保険金請求、税申告、給与明細など。各プリセットはキーワードと RAG 向け推奨クエリを含みます。

## RAG 連携 / RAG Integration / Intégration RAG
- `export_rag_bundle` が Markdown ダイジェスト、セル JSONL、テーブル別セクション、推奨クエリを `rag/manifest.json` にまとめます。
- サマリー (`summary` サブコマンド) から `rag_markdown` や `rag_suggested_queries` を取得し、下流エージェントに渡せます。
- `post_rag` フックでカスタム転送やストレージ連携を差し込めます。

## ビジュアライゼーション / Visualisation / Visualisation
- 4 パネルのマイクロスコープ（原画、シャープ、二値、勾配）と X-Ray オーバーレイを自動生成。
- `--views-log` で生成履歴を CSV 追記し、監視や QA に利用可能。

## サンプル素材 / Sample Assets / Ressources d'exemple
- `samples/demo_inputs/` にファイルを配置すると、`--input demo` がそれらを取り込みます。
- フォルダは空でも構いません。クローン直後は同梱の合成サンプルが利用されます。
- `samples/README.md` に多言語で手順をまとめています。

## 依存関係 / Dependencies / Dépendances
- Python 3.9+
- NumPy, Pillow, tqdm（コア機能）
- Numba（任意、BM25 DF の並列化）
- Poppler など PDF レンダラ（PDF 入力時に必要な場合あり）

## バックアップの単一ファイル版 / Single-file Backup / Fichier unique de secours
- 既存ワークフローが `zocr_allinone_merged_plus.py` に依存している場合、そのまま利用できます。
- モジュラー版と同じ CLI/サブコマンド/プラグイン/レポート機能を保持しています。
