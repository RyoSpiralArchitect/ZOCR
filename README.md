# Z-OCR All-in-One / Z-OCR オールインワン / Z-OCR tout-en-un

## 概要 / Overview / Aperçu
- **[JA]** 単一ファイル `zocr_allinone_merged_plus.py` が OCR から監視・学習までの処理をまるごと抱え、再現性と観測性を意識した運用を支援します。
- **[EN]** The single file `zocr_allinone_merged_plus.py` bundles OCR, augmentation, indexing, monitoring, and learning with reproducibility and observability in mind.
- **[FR]** Le fichier unique `zocr_allinone_merged_plus.py` réunit OCR, augmentation, indexation, surveillance et apprentissage en privilégiant la reproductibilité et l'observabilité.

- **[JA]** 同時に `zocr/` パッケージ（`consensus/`, `core/`, `orchestrator/`）として 3 分割したモジュールも収録し、既存インポート（`zocr_onefile_consensus` など）との互換性を維持したまま再利用できます。
- **[EN]** Alongside it, the repo now exposes a three-module `zocr/` package (`consensus/`, `core/`, `orchestrator/`) so projects can reuse the pieces while keeping backward-compatible import names such as `zocr_onefile_consensus`.
- **[FR]** En parallèle, un paquet `zocr/` en trois modules (`consensus/`, `core/`, `orchestrator/`) est fourni pour réutiliser chaque brique tout en conservant la compatibilité des imports historiques comme `zocr_onefile_consensus`。

## レイアウト / Layout / Structure
```
zocr/
  consensus/zocr_consensus.py    # OCR + table reconstruction
  core/zocr_core.py              # augment/index/query/monitor/sql
  orchestrator/zocr_pipeline.py  # pipeline orchestrator & watchdog
zocr_allinone_merged_plus.py     # backwards-compatible single-file bundle
```

## 使い方 / Usage / Utilisation
```bash
# 初回実行 / First run / Première exécution
python zocr_allinone_merged_plus.py --profile invoice_jp --snapshot --seed 12345

# 途中から再開 / Resume after failure / Reprendre après un échec
python zocr_allinone_merged_plus.py --resume --seed 12345

# インタラクティブ検索 / Interactive query / Recherche interactive
python zocr_allinone_merged_plus.py query --jsonl out/doc.mm.jsonl --index out/doc.bm25.pkl --q "total due"

# 履歴確認 / Inspect history / Consulter l'historique
python zocr_allinone_merged_plus.py history --outdir out_allinone --limit 10

# サマリー要約 / Summarise outputs / Résumer les sorties
python zocr_allinone_merged_plus.py summary --outdir out_allinone --keys sql_csv sql_schema monitor_csv

# プラグイン一覧 / List plugins / Lister les plugins
python zocr_allinone_merged_plus.py plugins
```

## 仕組み / Mechanics / Fonctionnement
- **[JA]** OCR → JSONL 書き出し → 多領域拡張 → BM25 インデックス → モニタリング → （任意で）パラメータ自動調整 → レポート更新、の順に `_safe_step` で保護された段階処理を走らせます。
- **[EN]** The pipeline executes OCR → JSONL export → multi-domain augmentation → BM25 indexing → monitoring → optional autotune → reporting, with each stage wrapped by `_safe_step` for logging and resumability.
- **[FR]** Le pipeline enchaîne OCR → export JSONL → augmentation multi-domaines → indexation BM25 → surveillance → autotuning optionnel → rapport, chaque étape étant protégée par `_safe_step` pour journalisation et reprise.

## 全体アーキテクチャ / Architecture / Architecture globale
```
入力 / Input / Entrée
    ↓
[OCR & Consensus]
    ↓
[Augment & Fusion]
    ↓
[BM25 + SQL Export]
    ↓
[Monitoring / Watchdog]
    ↓
[Tuning & Learning]
    ↓
出力 / Output / Sortie
```

## ビジュアライゼーション / Visualisation / Visualisation
- **[JA]** `views/` には 4 分割のマイクロスコープ画像（原画 ×3, エッジ強調, Otsu 二値, 勾配ヒートマップ）と、原画に疑似カラーの X-Ray オーバーレイを重ねたファイルを自動生成します。
- **[EN]** The `views/` folder now holds a four-panel microscope mosaic (raw ×3, edge sharpened, Otsu binarisation, gradient heatmap) plus an X-ray false-colour overlay blended with the original crop.
- **[FR]** Le dossier `views/` contient désormais une mosaïque microscope en quatre panneaux (brut ×3, renforcement des contours, binarisation Otsu, carte thermique de gradient) ainsi qu'une superposition faux-couleur de type rayon X appliquée à l'image d'origine.

## 対応ドメイン / Supported domains / Domaines pris en charge
- **[JA]** インボイス（日・英・仏）、見積書、納品書、領収書、契約書、購入注文書、経費精算、タイムシート、出荷案内、医療領収書などを検出・最適化用プリセットとして収録しています。
- **[EN]** Domain presets span invoices (JP / EN / FR), estimates, delivery slips, receipts, contracts, purchase orders, expenses, timesheets, shipping notices, and Japanese medical receipts for automatic detection and tuning.
- **[FR]** Les préréglages de domaine couvrent factures (JP / EN / FR), devis, bons de livraison, reçus, contrats, bons de commande, notes de frais, feuilles de temps, avis d'expédition et reçus médicaux japonais pour la détection et l'optimisation automatiques.

## 依存関係 / Dependencies / Dépendances
- Python 3.9+
- NumPy, Pillow, tqdm
- Numba (オプション / optional / optionnel) — 並列 BM25 DF 計算を高速化
- `pip install -r requirements.txt` がない環境では、上記パッケージを個別に導入してください。

## スナップショットとプラグイン / Snapshots & Plugins / Instantanés & Plugins
- `--snapshot` を付けると `pipeline_meta.json` にハッシュやバージョン情報を保存します。
- `--resume` により `pipeline_history.jsonl` を参照して途中段階から再開できます。
- `zocr_pipeline_allinone.register(stage)` でプラグインを登録し、`post_export` や `post_sql` などのフックでカスタム処理を差し込めます。
- `history` / `summary` / `plugins` サブコマンドで、実行履歴や生成物、登録済みフックを CLI から参照できます。

