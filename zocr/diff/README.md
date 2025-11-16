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
python -m zocr.diff \
  --a out/A \
  --b out/B \
  --out_diff out/diff/changes.diff \
  --out_json out/diff/events.json \
  --out_html out/diff/report.html
```

- **[JA]** `--a` / `--b` には `cells.jsonl` そのものか、`out/<run>/` のような実行ディレクトリを渡せます（後者は `rag/cells.jsonl` を自動解決）。
- **[EN]** `--a` / `--b` accept either explicit `cells.jsonl` files or a run directory such as `out/<run>/` (the CLI looks for `rag/cells.jsonl`).
- **[FR]** `--a` / `--b` peuvent pointer vers les fichiers `cells.jsonl` ou directement vers un dossier d’exécution (`out/<run>/`) ; la CLI y cherche `rag/cells.jsonl`.
- **[JA]** `--sections_a` / `--sections_b` を指定すると `sections.jsonl` のパスを明示でき、未指定時は `cells.jsonl` と同じ場所を自動探索します。ディレクトリを渡した場合も `rag/sections.jsonl` を解決します。
- **[EN]** Optional `--sections_a` / `--sections_b` flags override auto-discovery of `sections.jsonl`; directories are resolved to `rag/sections.jsonl` just like the cell inputs.
- **[FR]** Les options `--sections_a` / `--sections_b` permettent de fournir explicitement les chemins `sections.jsonl`; lorsqu’un dossier est fourni, la CLI y cherche `rag/sections.jsonl` automatiquement.
- **[JA]** `--out_plan` で差分イベントを再解析キュー / RAG 補助 / プロファイル更新に分類した `assist_plan.json` を保存し、既存の請求書向けフィードバックループへ即連携できます。
- **[EN]** `--out_plan` writes an `assist_plan.json` that splits the diff feed into reanalysis queues, downstream RAG follow-ups, and profile tweaks so the invoice-domain loops can reuse it directly.
- **[FR]** `--out_plan` génère un `assist_plan.json` qui classe les événements (réanalyse, suivi RAG, ajustements de profil) pour alimenter directement les boucles déjà en service sur le domaine facturation.
- **[JA]** `--simple_text_a` / `--simple_text_b` を指定すると ToyOCR などのプレーンテキスト比較に適した軽量 differ が有効になり、git 風 unified diff と金額/数量の数値差分を `--simple_diff_out` / `--simple_json_out` で保存できます（`--simple_plan_out` を付ければ再解析/RAG 補助バンドルも同時に書き出し）。
- **[EN]** Supplying `--simple_text_a` / `--simple_text_b` toggles the ToyOCR-friendly quick differ, producing a git-like unified diff plus amount/quantity deltas that can be persisted via `--simple_diff_out` / `--simple_json_out`; add `--simple_plan_out` to save the matching reanalysis/RAG assist bundle.
- **[FR]** Avec `--simple_text_a` / `--simple_text_b`, on active le diff léger compatible ToyOCR, lequel exporte un diff unifié façon git et les deltas montants/quantités via `--simple_diff_out` / `--simple_json_out`, tandis que `--simple_plan_out` produit en plus le bundle d’assistance réanalyse/RAG.
- **[JA]** 行の追加・削除も監視し、置換以外の差分（例：費目が 1 行だけ増えた請求書）でも該当金額の Δ/率を抽出します。
- **[EN]** Inserted/deleted lines are covered alongside replacements, so a newly added fee line still yields the precise Δ/relative delta.
- **[FR]** Les lignes ajoutées/supprimées sont également prises en compte, ce qui permet d’extraire Δ/variation même quand une seule ligne vient s’ajouter.
- **[JA]** simple 入力にも `cells.jsonl` と同様にディレクトリを渡せ、`rag/bundle.md` → `bundle.md` → `preview.md` → `.txt` の順に探索するため、ToyOCR/ZOCR の RAG 出力フォルダを指すだけで比較が走ります。
- **[EN]** Like the semantic inputs, the simple mode accepts run directories: the resolver scans `rag/bundle.md`, `bundle.md`, `preview.md`, then `.txt` variants so pointing at a ToyOCR/ZOCR run folder is enough.
- **[FR]** À l’instar des entrées sémantiques, le mode léger accepte un dossier d’exécution : il y cherche successivement `rag/bundle.md`, `bundle.md`, `preview.md` puis les variantes `.txt`, ce qui suffit pour comparer un dossier ToyOCR/ZOCR tel quel.
- **[JA]** セマンティック diff を実行しない場合は `--a` / `--b` を空のままにし、上記の simple フラグと出力先のみ指定すれば軽量モード単体で完結します（`--out_*` や `--sections_*` は不要）。
- **[EN]** To skip the semantic pass entirely, simply omit `--a` / `--b` and provide only the simple flags plus their outputs; no semantic `--out_*` / `--sections_*` parameters are required.
- **[FR]** Pour sauter le diff sémantique, ne renseignez pas `--a` / `--b` et fournissez uniquement les options du mode léger avec leurs sorties ; nul besoin d’ajouter les paramètres `--out_*` / `--sections_*` du mode principal.

### Assist plan / アシストプラン / Plan d’assistance
- **[JA]** `assist_plan.json` は `reanalyze_queue` / `rag_followups` / `profile_actions` を含み、各エントリに行プレビューや `trace_id` を付与するため、Slack/Teams 通知や `intent.action="reanalyze_cells"` トリガにそのまま使えます。
- **[EN]** `assist_plan.json` groups recommendations into `reanalyze_queue`, `rag_followups`, and `profile_actions` while preserving row previews plus `trace_id`s so it can feed Slack/Teams digests or fire `intent.action="reanalyze_cells"` automatically.
- **[FR]** `assist_plan.json` regroupe les recommandations (`reanalyze_queue`, `rag_followups`, `profile_actions`) avec aperçus de lignes et `trace_id`, prêt à déclencher `intent.action="reanalyze_cells"` ou à nourrir des notifications Slack/Teams.
- **[JA]** さらに `domain_tags` / `llm_directive` / `domain_briefings` / `handoff_packets` があり、請求書・契約・物流だけでなく医療・保険・製造・エネルギー・コンプラ・不動産・通信・小売・官公庁・教育・テック・マーケ・航空・建設にも対応した diff テンプレを生成します。
- **[EN]** Each entry also exposes `domain_tags`, an LLM-oriented `llm_directive`, aggregated `domain_briefings`, and consolidated `handoff_packets`, covering invoice/contract/logistics plus healthcare/insurance/manufacturing/energy/compliance as well as real-estate/telecom/retail/public-sector/education/technology/marketing/aviation/construction use cases.
- **[FR]** Chaque entrée inclut désormais `domain_tags`, une `llm_directive`, des `domain_briefings` et des `handoff_packets`, afin de servir les domaines facture/contrat/logistique, santé/assurance/fabrication/énergie/conformité mais aussi immobilier/télécom/retail/secteur public/éducation/technologie/marketing/aviation/construction.

- **[JA]** エントリには `llm_ready_context` / `handoff_brief` も追加され、`handoff_packets` は `llm_context_examples` を添付して差分の位置・理由・旧新値をまとめたプロンプトをそのまま LLM へ渡せます。
- **[EN]** Entries ship with `llm_ready_context` plus a concise `handoff_brief`, and each packet now lists `llm_context_examples` so downstream LLMs inherit a diff-specific prompt with the location, rationale, and before/after values.
- **[FR]** Chaque entrée fournit `llm_ready_context` et un `handoff_brief` concis, tandis que les paquets incluent `llm_context_examples`, donnant aux LLM aval un prompt diff prêt à l’emploi (emplacement, raison, valeurs avant/après).

## Quick git-style differ / 軽量 git 風 diff / Diff git simplifié
- **[JA]** `SimpleTextDiffer` は 2 つのプレーンテキスト（ToyOCR の出力や編集済み仕様書など）を読み、git と同じ unified diff を生成しつつ、行ごとの金額・数量の差分（Δ/相対率）を JSON に書き出します。置換だけでなく行の追加・削除（例：費目を 1 行追加／削除）でも Δ/率を抽出します。
- **[EN]** `SimpleTextDiffer` lets you compare two plain-text documents (ToyOCR dumps, spec revisions, memo drafts) with a git-style unified diff plus structured numeric deltas (absolute + relative) per changed line, and it now surfaces inserted/deleted rows in addition to replacements (e.g., a brand-new line item).
- **[FR]** `SimpleTextDiffer` compare deux documents texte (exports ToyOCR, spécifications modifiées, brouillons) en produisant un diff unifié façon git et des deltas numériques structurés (absolu + relatif) par ligne, tout en détectant désormais les lignes ajoutées/supprimées au même titre que les remplacements (nouveau poste, suppression d’une ligne).

```bash
python -m zocr.diff \
  --a out/A --b out/B \
  --simple_text_a memo_v1.txt \
  --simple_text_b memo_v2.txt \
  --simple_diff_out out/diff/memo.diff \
  --simple_json_out out/diff/memo.numeric.json \
  --simple_plan_out out/diff/memo.assist.json
```

```bash
# semantic diff を省き、軽量モードのみ実行する例
# Example showing the quick differ on its own / Exemple en mode léger seul
python -m zocr.diff \
  --simple_text_a memo_v1.txt \
  --simple_text_b memo_v2.txt \
  --simple_diff_out out/diff/memo.diff \
  --simple_json_out out/diff/memo.numeric.json \
  --simple_plan_out out/diff/memo.assist.json
```

- **[JA]** 同一フォーマットで数値だけ揺れる社内帳票や見積書を git diff そのままの感覚で比較し、差分のうち数値が変わった箇所を Slack などに流すだけならこのモードで完結します。
- **[EN]** When two revisions share almost identical wording/layout and you just need the amount/quantity changes (e.g., ToyOCR exports, estimates, order forms), this mode provides the entire answer without loading the heavier semantic differ.
- **[FR]** Pour des documents quasi identiques (factures, devis, formulaires) où seules les valeurs changent, ce mode suffit : diff unifié + détection des montants modifiés, prêt pour les notifications Slack/Teams.
- **[JA]** `SimpleTextDiffer.events_from_result` は数値差分を `cell_updated` イベントに変換し、`DiffAssistPlanner` と同じ分類 (`reanalyze_queue` / `rag_followups` / `profile_actions`)・`handoff_packets`・`llm_ready_context` を生成するため、`simple_plan_out` で保存した JSON を下流 RAG や補助依頼にそのまま渡せます。
- **[EN]** `SimpleTextDiffer.events_from_result` converts each numeric delta into a `cell_updated` event, letting the same `DiffAssistPlanner` derive `reanalyze_queue` / `rag_followups` / `profile_actions` plus `handoff_packets` and `llm_ready_context`; the `simple_plan_out` artifact is therefore ready for downstream RAG/support queues.
- **[FR]** `SimpleTextDiffer.events_from_result` transforme les deltas numériques en événements `cell_updated`, ce qui permet à `DiffAssistPlanner` de calculer les mêmes files (`reanalyze_queue`, `rag_followups`, `profile_actions`) ainsi que `handoff_packets` et `llm_ready_context`; le fichier produit via `simple_plan_out` peut donc alimenter immédiatement les équipes RAG/assistance.

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

## Frontier significance / フロンティアとしての意義 / Portée de la frontière
- **[JA]** まだ市場には「請求書・法務文書・CAD 図表・営業仕様書を意味構造ごと比較できる diff」がありません。`zocr.diff` は `cells.jsonl` / `sections.jsonl` を活かし、表・節・filters を束ねて frontier を押さえることで Z-OCR 全体の技術的アイデンティティを確立します。
- **[EN]** No production tool currently performs semantic diffs for invoices, legal docs, CAD-like grids, and shifting business specs. By leaning on the existing bundle, `zocr.diff` owns that frontier and turns Z-OCR into the platform that names and tracks those structural deltas.
- **[FR]** Le marché ne propose pas encore de diff sémantique couvrant factures, documents juridiques, tableaux CAD ou spécifications métier évolutives. En capitalisant sur le bundle existant, `zocr.diff` occupe cette frontière et fait de Z-OCR la plateforme qui identifie ces écarts structurels.

### Downstream loops / 下流ループ連携 / Boucles aval
- **[JA]** 生成されたイベントは `orchestrator` の monitor / intent / reanalysis ループ（請求書ドメインで実績済み）へそのまま流せます。差分イベントを Slack/Teams 通知に回し、必要に応じて `intent.action="reanalyze_cells"` を自動でリクエストすれば、中流の補助・再解析チームにもワンクリックで依頼できます。
- **[EN]** You can pipe the events directly into the orchestrator’s monitor, intent, and reanalysis loops that already power invoice-domain reruns. Dispatching the diff feed to Slack/Teams plus auto-requesting `intent.action="reanalyze_cells"` lets downstream mid-stream assistants pick up tasks without additional plumbing.
- **[FR]** Les événements se branchent directement sur les boucles monitor/intent/réanalyse de l’orchestrateur, déjà éprouvées côté factures. Il suffit d’alimenter Slack/Teams et de demander automatiquement `intent.action="reanalyze_cells"` pour que les équipes intermédiaires prennent le relais sans travail supplémentaire.
