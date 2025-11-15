# Semantic Diff Module / セマンティック差分モジュール / Module de diff sémantique

## 概要 / Overview / Aperçu
- **[JA]** `zocr.diff` は RAG バンドル（`cells.jsonl` / `sections.jsonl`）を直接読み取り、テーブル/セクション単位でマッチングした構造化イベントとテキスト/HTMLレポートを生成します。
- **[EN]** `zocr.diff` ingests the existing RAG bundle artifacts (`cells.jsonl`, `sections.jsonl`) to align tables and sections between two runs and emit structured events plus unified-text/HTML reports.
- **[FR]** `zocr.diff` consomme directement les artefacts du bundle RAG (`cells.jsonl`, `sections.jsonl`) pour apparier tableaux/sections entre deux exécutions et produire des événements structurés ainsi que des rapports texte/HTML.

## ディレクトリ / Directory layout / Arborescence
```
zocr/diff/
 ├─ __init__.py        # public surface / 公開API / surface publique
 ├─ differ.py          # matching + events / マッチングとイベント / appariement + événements
 ├─ render.py          # renderers / レンダラ / moteurs de rendu
 └─ cli.py             # CLI wrapper / CLI / enveloppe CLI
```

## 入力要件 / Input expectations / Entrées attendues
- **`cells.jsonl`**
  - **[JA]** 必須。`page` / `table_index` / `row` / `col` / `text` / `filters` / `trace_id` などを持つセル記録。
  - **[EN]** Required. Structured cell rows carrying `page`, `table_index`, `row`, `col`, `text`, `filters`, `trace_id`, and optional header hints.
  - **[FR]** Obligatoire. Enregistrements structurés avec `page`, `table_index`, `row`, `col`, `text`, `filters`, `trace_id` et, si présent, un en-tête.
- **`sections.jsonl`**
  - **[JA]** 任意。章・節の見出しを使ってヘッダーレベルの差分イベントを生成。
  - **[EN]** Optional. Provides heading metadata so the differ can report section-level changes.
  - **[FR]** Optionnel. Apporte les titres de sections pour signaler les deltas au niveau des rubriques.

両ファイルとも `out/<run>/rag/` に既定で出力されるため、2 回の実行結果をすぐに突き合わせられます。

Both files live under `out/<run>/rag/`, letting you compare two runs immediately after they finish.

Les deux fichiers sont exportés dans `out/<run>/rag/`, ce qui permet de comparer deux exécutions aussitôt terminées.

## 組み込み CLI / Built-in CLI / CLI intégrée
```bash
python -m zocr.diff.cli \
  --a out/A/rag/cells.jsonl \
  --b out/B/rag/cells.jsonl \
  --out_diff out/diff/changes.diff \
  --out_json out/diff/events.json \
  --out_html out/diff/report.html
```

- **[JA]** `--sections_a` / `--sections_b` を指定すると `sections.jsonl` のパスを明示でき、未指定時は `cells.jsonl` と同じ場所を自動探索します。
- **[EN]** Optional `--sections_a` / `--sections_b` flags override auto-discovery of `sections.jsonl` next to each `cells.jsonl`.
- **[FR]** Les options `--sections_a` / `--sections_b` permettent de fournir explicitement les chemins `sections.jsonl`; sinon, ils sont détectés automatiquement à côté de chaque `cells.jsonl`.

## なぜ小さく保てるか / Why the implementation stays small / Pourquoi si peu de code suffit
1. **構造化セル情報 / Structured cell context / Contexte cellulaire structuré** – 各レコードにページ・表・行列・テキスト・filters・`trace_id` が揃っているため、再OCRではなく構造合わせに集中できます。
2. **RAG バンドルの再利用 / Ready-made RAG bundle / Bundle RAG prêt à l’emploi** – `cells.jsonl` / `sections.jsonl` / table サマリ / Markdown プレビューなどが既に揃い、A/B 比較が即可能。
3. **trace_id / Trace IDs / Identifiants de trace** – `doc=A;page=1;table=0;row=3;col=2` のような由来情報が、列並べ替えや行順変更後も同値性を保つ軸になります。
4. **フィルター由来の意味情報 / Normalised semantic hints / Indices sémantiques normalisés** – `filters.amount` / `filters.qty` / `filters.date` / `synthesis_window` / ベクトル特徴量などが数値比較や文脈判断を後押し。
5. **フィードバックループの足場 / Feedback-loop hooks / Boucles de rétroaction déjà en place** – monitor / reanalysis / autotune などが差分イベントをすぐに監視・通知パイプラインへ流し込めます。
6. **JSONL 供給 / JSONL inputs / Entrées JSONL** – `out/A/rag/cells.jsonl` と `out/B/rag/cells.jsonl` を指すだけで、十数行のオーケストレーションでセマンティック diff が走ります。

これらの部品が揃っているため、`zocr.diff` はマッチング手法・イベントスキーマ・レンダラの洗練に集中でき、追加のエクスポーターや特殊ログを用意する必要がありません。

Because these ingredients already exist, `zocr.diff` can focus on matching heuristics, event schemas, and renderers without inventing bespoke exporters or logging layers.

Grâce à ces briques existantes, `zocr.diff` se concentre sur les heuristiques d’appariement, les schémas d’événements et les rendus, sans exiger d’exportateurs ni de journaux sur mesure.
