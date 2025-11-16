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
- **[JA]** `--out_agentic` を付けると AgenticRAG/GUI 連携向けに `agentic_requests.json` も書き出され、各イベントへ JA/EN 説明用プロンプトと差分画像の描画指示がセットになります。
- **[EN]** Add `--out_agentic` to export `agentic_requests.json`, a bundle of bilingual directives plus visual-overlay briefs so Agentic RAG/GUI helpers can craft diff images or narrated summaries without reprocessing the events.
- **[FR]** L’option `--out_agentic` produit `agentic_requests.json`, c’est-à-dire des requêtes bilingues accompagnées d’instructions de rendu pour que les agents RAG/GUI génèrent images ou explications immédiatement.
- **[JA]** `--out_markdown` を指定すると `report.md` が得られ、Slack/Teams 共有やナレッジベース貼付が容易になります（orchestrator の `run_diff` も同ファイルを出力し、`handoff_bundle.json` には `markdown_report` として本文を内包します）。
- **[EN]** Use `--out_markdown` to emit `report.md`, a shareable Markdown digest for chats/dashboards; the orchestrator’s `run_diff` writes the same file and `handoff_bundle.json` embeds the text via `markdown_report`.
- **[FR]** Avec `--out_markdown`, on produit `report.md`, un résumé Markdown prêt pour Slack/Teams ou la documentation ; `run_diff` génère le même fichier et `handoff_bundle.json` embarque ce texte dans `markdown_report`.
- **[JA]** `--out_bundle` を指定するとイベント / unified diff / Markdown レポート / assist plan / agentic requests を丸ごと含む `handoff_bundle.json` が生成され、API や GUI から 1 ファイルで取得できます（orchestrator の `run_diff` も同ファイルを出力）。
- **[EN]** Pass `--out_bundle` to produce `handoff_bundle.json`, a single artifact carrying the events, unified diff text, Markdown digest, assist plan, and agentic requests so APIs/GUI surfaces can ingest the entire package at once (the orchestrator’s `run_diff` writes the same file).
- **[FR]** Avec `--out_bundle`, on obtient `handoff_bundle.json`, un unique artefact regroupant événements, diff texte, résumé Markdown, plan d’assistance et requêtes agentic afin que l’API/la GUI récupère tout d’un coup (le `run_diff` de l’orchestrateur le génère aussi).
- **[JA]** 差分対象文書の枚数制限を撤廃し、`DiffAssistPlanner` のバケットも無制限化したので、複数案件を 1 回の比較に詰め込んでもイベントが欠けません。軽量 differ は RapidFuzz ベースのシグネチャをキャッシュするため、ToyOCR テキストでも数十ページ分を俊敏にマッチングできます。
- **[EN]** Document caps are gone—the `DiffAssistPlanner` buckets no longer clip entries, so large batches survive intact. The lightweight differ caches RapidFuzz-derived signatures, keeping ToyOCR/text comparisons fast even when you feed it dozens of pages.
- **[FR]** La limite sur le nombre de documents comparés a été supprimée : les compartiments du `DiffAssistPlanner` n’élaguent plus rien, même pour de gros lots. Le diff léger met en cache des signatures issues de RapidFuzz afin de conserver des performances élevées sur des dizaines de pages ou exports ToyOCR.
- **[JA]** `summary.numeric_summary` には通貨/単位ごとのバケット、総和、最大差トップ5が入り、`assist_plan.json` や handoff bundle だけ見ても「USD +12,000 / % -3pt」のような勘所が即分かります（軽量 diff も同じ構造）。
- **[EN]** `summary.numeric_summary` now aggregates the net/absolute totals, per-currency or per-unit buckets, and the five largest swings so even a dashboard or handoff bundle alone can highlight “USD +12,000 / % -3pt”; the quick differ produces the same payload.
- **[FR]** `summary.numeric_summary` regroupe désormais totaux nets/absolus, compartiments par devise/unité et les cinq plus fortes variations afin que les bundles ou tableaux de bord annoncent immédiatement « USD +12 000 / % −3 pts » ; le diff léger fournit la même structure.
- **[JA]** `summary.textual_summary` では `text_change_type` ごとの件数・平均類似度/overlap/Jaccard・頻出トークンと代表例をまとめるため、備考や条文の追記も bundle 単体で共有できます（軽量 diff も同じフィールド）。
- **[EN]** `summary.textual_summary` captures counts per `text_change_type`, the average similarity/overlap/jaccard, highlighted tokens, and the most notable rewrites so commentary updates are visible even if you only open the bundle (the lightweight diff emits the same field).
- **[FR]** `summary.textual_summary` fournit les volumes par `text_change_type`, les similarités/overlaps/jaccards moyens, les tokens marquants et les principales réécritures, ce qui rend les annotations ou clauses mises à jour visibles directement depuis le bundle (même structure côté diff léger).
- **[JA]** `summary.section_summary` は章・節ごとのイベント件数、数値/テキスト比率、累積差額、代表差分を並べるため、`report.md` や `handoff_bundle.json` だけで「Section 4 が +€9k / テキスト差分 3 件」と即座に共有できます。
- **[EN]** `summary.section_summary` tracks per-section counts, numeric/textual splits, cumulative deltas, and representative rows so `report.md` or `handoff_bundle.json` can report “Section 4: +€9k / 3 textual edits” without reopening the PDFs.
- **[FR]** `summary.section_summary` recense pour chaque section le volume d'événements, la part numérique/texte, le delta cumulé et quelques exemples, de sorte que `report.md` ou `handoff_bundle.json` suffisent à annoncer « Section 4 : +9 k€ / 3 révisions textuelles ».
- **[JA]** `assist_plan.json` には `impact_summary` が加わり、総スコア・バケット別件数・トップエントリがまとまります。各 recommendation と agentic request にも `impact_score` / `impact_bucket` が入るため、GUI や API から重要度順のソートが可能です。
- **[EN]** `assist_plan.json` now includes an `impact_summary` (total score, bucket counts, top entries) while every recommendation and agentic request exposes `impact_score` / `impact_bucket`, making it easy for GUIs/APIs to sort the feed by urgency.
- **[FR]** `assist_plan.json` comporte désormais `impact_summary` (score total, compte par compartiment, meilleures entrées) et chaque recommandation/requête agentic fournit `impact_score` / `impact_bucket`, ce qui permet de trier le flux selon l’importance côté GUI/API.
- **[JA]** `--simple_text_a` / `--simple_text_b` を指定すると ToyOCR などのプレーンテキスト比較に適した軽量 differ が有効になり、git 風 unified diff と金額/数量の数値差分を `--simple_diff_out` / `--simple_json_out` で保存できます（`--simple_plan_out` を付ければ再解析/RAG 補助バンドルも同時に書き出し）。
- **[EN]** Supplying `--simple_text_a` / `--simple_text_b` toggles the ToyOCR-friendly quick differ, producing a git-like unified diff plus amount/quantity deltas that can be persisted via `--simple_diff_out` / `--simple_json_out`; add `--simple_plan_out` to save the matching reanalysis/RAG assist bundle.
- **[FR]** Avec `--simple_text_a` / `--simple_text_b`, on active le diff léger compatible ToyOCR, lequel exporte un diff unifié façon git et les deltas montants/quantités via `--simple_diff_out` / `--simple_json_out`, tandis que `--simple_plan_out` produit en plus le bundle d’assistance réanalyse/RAG.
- **[JA]** 軽量 differ 単体で動かす場合も `--simple_agentic_out` を併用すれば `agentic_requests.json` を個別に保存でき、ToyOCR 由来のイベントをそのまま AgenticRAG の差分画像/説明タスクへ引き継げます。
- **[EN]** When relying only on the quick differ, pass `--simple_agentic_out` to persist its own `agentic_requests.json`, keeping ToyOCR events compatible with the same Agentic RAG workflows.
- **[FR]** Pour le diff léger seul, `--simple_agentic_out` exporte aussi `agentic_requests.json`, garantissant une compatibilité immédiate avec les agents RAG/GUI qui produisent images ou narratifs.
- **[JA]** `--simple_bundle_out` を使うと軽量 differ 版の `handoff_bundle.json` も残せるため、ToyOCR で拾った差分と Markdown サマリを API / GUI / AgenticRAG へワンショットで手渡せます。
- **[EN]** Use `--simple_bundle_out` to save the quick-differ `handoff_bundle.json`, allowing ToyOCR-style comparisons (plus the Markdown digest) to feed APIs, GUI cards, or Agentic RAG actors from one payload.
- **[FR]** `--simple_bundle_out` conserve la variante légère de `handoff_bundle.json`, désormais accompagnée du résumé Markdown pour livrer aux API/GUI ou aux agents RAG l’ensemble du paquet en une seule fois.
- **[JA]** `--simple_markdown_out` で軽量 diff 専用の `report.md` も得られるため、git 風 diff を開かなくても差分の概要を共有できます。
- **[EN]** Add `--simple_markdown_out` to capture a quick-differ `report.md`, letting you paste a chat-friendly summary without reopening the raw git-style diff.
- **[FR]** En ajoutant `--simple_markdown_out`, le diff léger exporte également `report.md`, un résumé partageable sans rouvrir le diff brut.
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
- **[JA]** 同ファイル内の `agentic_requests` には `visual_brief` / `narrative_brief` / `handoff_focus` / `preferred_outputs` が並び、差分画像生成・自然言語説明・GUI カード化といった AgenticRAG タスクがそのまま起動できるようになっています。
- **[EN]** The embedded `agentic_requests` array captures `visual_brief`, `narrative_brief`, `handoff_focus`, and `preferred_outputs`, letting Agentic RAG workers produce overlays, GUI cards, or bilingual explanations straight from the diff feed.
- **[FR]** Le tableau `agentic_requests` fournit `visual_brief`, `narrative_brief`, `handoff_focus` et `preferred_outputs`, ce qui permet aux agents RAG/GUI de générer immédiatement images de diff, cartes GUI ou explications bilingues.
- **[JA]** さらに `domain_tags` / `llm_directive` / `domain_briefings` / `handoff_packets` があり、請求書・契約・物流に加えて医療・保険・製造・エネルギー・コンプラ・不動産・通信・小売・官公庁・教育・テック・マーケ・航空・建設だけでなく、自動車・ホスピタリティ・メディア・銀行・ゲーム・飲食・農業・鉱業・海運・スポーツ・エンタメ・非営利・サイバーセキュリティ案件にも専用テンプレを生成します。
- **[EN]** Each entry also exposes `domain_tags`, an LLM-oriented `llm_directive`, aggregated `domain_briefings`, and consolidated `handoff_packets`, covering invoice/contract/logistics plus healthcare/insurance/manufacturing/energy/compliance, real-estate/telecom/retail/public-sector/education/technology/marketing/aviation/construction, and now the expanded automotive/hospitality/media/banking/gaming/food-beverage/agriculture/mining/shipping/sports/entertainment/nonprofit/cybersecurity verticals.
- **[FR]** Chaque entrée inclut `domain_tags`, une `llm_directive`, des `domain_briefings` et des `handoff_packets`, couvrant factures/contrats/logistique, santé/assurance/fabrication/énergie/conformité, immobilier/télécom/retail/secteur public/éducation/technologie/marketing/aviation/construction, ainsi que les nouveaux domaines automobile/hôtellerie/médias/banque/jeu vidéo/restauration-agro/agriculture/mines/shipping/sports/entertainement/organisations à but non lucratif/cybersécurité.
- **[JA]** 各エントリには `a_row_context` / `b_row_context` / `row_context_radius` や `line_signature` / `text_highlight` / `text_token_stats` などの軽量 diff メタデータも複製され、原文を開かなくても周辺文脈とハイライトを参照できます。
- **[EN]** Entries also replicate the lightweight diff metadata—`a_row_context`, `b_row_context`, `row_context_radius`, plus `line_signature`, `text_highlight`, `text_token_stats`, etc.—so downstream reviewers can inspect the nearby text and inline highlights without reopening the source files.
- **[FR]** Les entrées reprennent également les métadonnées du diff léger (`a_row_context`, `b_row_context`, `row_context_radius`, ainsi que `line_signature`, `text_highlight`, `text_token_stats`, etc.), ce qui permet aux réviseurs de visualiser le contexte et les surlignages sans rouvrir les documents.
- **[JA]** すべての `cell_updated` イベントはテキスト類似度と数値差から算出した 0〜1 の `confidence` を持ち、`report.md` / HTML / `assist_plan.json` / `agentic_requests.json` / handoff bundle にそのまま連携されます。
- **[EN]** Every `cell_updated` event carries a 0–1 `confidence` value derived from the text gap plus relative/numeric deltas, and the score is propagated to `report.md`, the HTML view, `assist_plan.json`, `agentic_requests.json`, and the handoff bundle.
- **[FR]** Chaque événement `cell_updated` inclut un score `confidence` (0 à 1) issu de l’écart textuel et des deltas relatifs/numériques, score qui se retrouve tel quel dans `report.md`, le rapport HTML, `assist_plan.json`, `agentic_requests.json` et le bundle d’handoff.
- **[JA]** `summary.impact_score_total` / `impact_bucket_counts` に加え、`impact_summary.top_entries` と各エントリの `impact_score` / `impact_bucket` で重要度を定量化でき、最もクリティカルな差分をダッシュボードで即座にハイライトできます。
- **[EN]** `summary.impact_score_total`, `impact_bucket_counts`, and `impact_summary.top_entries` quantify the feed, and every entry exposes `impact_score` / `impact_bucket`, so dashboards can instantly spotlight the most critical changes.
- **[FR]** `summary.impact_score_total`, `impact_bucket_counts` et `impact_summary.top_entries` quantifient désormais le flux, chaque entrée apportant `impact_score` / `impact_bucket` pour mettre en avant les écarts critiques dans les tableaux de bord.
- **[JA]** Markdown の `#` / `##`・Setext 見出しや `第3条` / `Section 2` といった条番号も解析し、`section_heading` / `section_path` / `section_level`（および `_a` / `_b`）を各エントリへコピーするため、ToyOCR 由来のメモ差分でも「どの章節で起きた変更か」を即共有できます。
- **[EN]** Markdown hashes/underlines plus legal-style headings such as `第3条` or `Section 2` are detected and copied into each entry as `section_heading`, `section_path`, and `section_level` (with side-specific `_a` / `_b` fields) so ToyOCR quick-diff output still tells downstream teams exactly which section changed.
- **[FR]** Les titres Markdown (`#`, `##`, variantes Setext) et les en-têtes juridiques (`第3条`, `Section 2`, etc.) sont repérés et injectés dans chaque entrée via `section_heading`, `section_path`, `section_level` (ainsi que les versions `_a` / `_b`), ce qui situe immédiatement la section concernée dans les diff ToyOCR.

- **[JA]** エントリには `llm_ready_context` / `handoff_brief` も追加され、`handoff_packets` は `llm_context_examples` を添付して差分の位置・理由・旧新値をまとめたプロンプトをそのまま LLM へ渡せます。
- **[EN]** Entries ship with `llm_ready_context` plus a concise `handoff_brief`, and each packet now lists `llm_context_examples` so downstream LLMs inherit a diff-specific prompt with the location, rationale, and before/after values.
- **[FR]** Chaque entrée fournit `llm_ready_context` et un `handoff_brief` concis, tandis que les paquets incluent `llm_context_examples`, donnant aux LLM aval un prompt diff prêt à l’emploi (emplacement, raison, valeurs avant/après).

## Quick git-style differ / 軽量 git 風 diff / Diff git simplifié
- **[JA]** `SimpleTextDiffer` は 2 つのプレーンテキスト（ToyOCR の出力や編集済み仕様書など）を読み、git と同じ unified diff を生成しつつ、行ごとの金額・数量の差分（Δ/相対率）を JSON に書き出します。置換だけでなく行の追加・削除（例：費目を 1 行追加／削除）でも Δ/率を抽出します。
- **[EN]** `SimpleTextDiffer` lets you compare two plain-text documents (ToyOCR dumps, spec revisions, memo drafts) with a git-style unified diff plus structured numeric deltas (absolute + relative) per changed line, and it now surfaces inserted/deleted rows in addition to replacements (e.g., a brand-new line item).
- **[FR]** `SimpleTextDiffer` compare deux documents texte (exports ToyOCR, spécifications modifiées, brouillons) en produisant un diff unifié façon git et des deltas numériques structurés (absolu + relatif) par ligne, tout en détectant désormais les lignes ajoutées/supprimées au même titre que les remplacements (nouveau poste, suppression d’une ligne).
- **[JA]** 数値が出てこない文章差分でも `textual_changes` を生成し、`text_change_type` / `line_similarity` に加えて `text_token_stats`（追加/削除/共通トークン + overlap/jaccard）と `text_highlight`（`[[…]]` マーカー付きのミニ差分）を `assist_plan.json` へ渡すため、「備考の追記」「条文差し替え」なども即座に把握できます。
- **[EN]** Text-only adjustments still emit `textual_changes`, but now each entry also carries `text_token_stats` (added/removed/common tokens plus overlap/jaccard) and `text_highlight` snippets that wrap edits with `[[…]]`, so downstream queues see the rewrite type via `text_change_type` / `line_similarity` *and* the exact phrasing delta inside `assist_plan.json`.
- **[FR]** Même pour les modifications purement textuelles, chaque entrée `textual_changes` transmet `text_change_type`, `line_similarity`, ainsi que `text_token_stats` (tokens ajoutés/supprimés/communs + overlap/jaccard) et `text_highlight` (aperçu `[[…]]`), ce qui permet aux agents aval de comprendre immédiatement la nature et le contenu de la réécriture.
- **[JA]** 行マッチングは `<num>` 正規化でノイズを除き、スペース揺れやラベル差があっても同一行として扱います。結果 JSON には `line_signature` / `line_label` / `line_similarity` が含まれ、ToyOCR → RAG のハンドオフでそのまま参照できます。
- **[EN]** Line pairing uses `<num>`-normalized fuzzy matches so even spacing tweaks or slight label edits keep pointing to the same row. The JSON now exposes `line_signature`, `line_label`, and `line_similarity`, which downstream ToyOCR/RAG agents can consume immediately.
- **[FR]** L’appariement des lignes s’appuie sur une normalisation `<num>` afin de rester robuste aux variations d’espaces ou aux légers changements d’intitulés. Le JSON inclut `line_signature`, `line_label` et `line_similarity`, directement exploitables par les agents ToyOCR/RAG.
- **[JA]** 数値を含まない備考や文章の差し替えでも `textual_changes` が出力され、`text_change_type` / `line_similarity` に加えて `text_token_stats` と `text_highlight` が付属するため、どのトークンが増減し、どんな文章に置き換わったのかを即座に共有できます。
- **[EN]** Even for prose-only edits, the quick differ annotates each `textual_changes` row with `text_change_type`, `line_similarity`, `text_token_stats`, and a `text_highlight` snippet so downstream tooling knows whether text was added/removed/rewritten *and* which exact tokens shifted.
- **[FR]** Les changements purement textuels fournissent `text_change_type`, `line_similarity`, mais aussi `text_token_stats` et `text_highlight`, ce qui indique quelles expressions ont été ajoutées/supprimées et comment la phrase a été réécrite.
- **[JA]** 通貨記号や USD/EUR/円 のような通貨コード、`千` / `万` / `億` / `k` / `M` / `B`、括弧付きマイナス、% / ％ を含む行でも金額/率を正しく抽出し、`numeric_unit` / `numeric_currency` / `numeric_is_percent` / `numeric_scale` でハンドオフ先に単位を共有します。
- **[EN]** Currency symbols/codes plus Japanese `千` / `万` / `億` and western `k` / `M` / `B`, parenthetical negatives, and %/％ tokens are parsed in-place; the emitted events include `numeric_unit`, `numeric_currency`, `numeric_is_percent`, and `numeric_scale` so downstream ToyOCR/RAG flows keep the unit context.
- **[FR]** Les symboles/codes de devise, les suffixes `千` / `万` / `億` ainsi que `k` / `M` / `B`, les montants négatifs entre parenthèses et les %/％ sont gérés automatiquement ; les événements exposent `numeric_unit`, `numeric_currency`, `numeric_is_percent` et `numeric_scale` pour préserver les unités dans les flux ToyOCR/RAG.
- **[JA]** さらに `a_row_context` / `b_row_context` / `row_context_radius` で変更行の前後抜粋も保持され、差分周辺テキストを補助ワークフローがすぐ参照できます。
- **[EN]** The quick differ also emits `a_row_context`, `b_row_context`, and `row_context_radius`, giving downstream helpers trimmed neighbor lines around each change.
- **[FR]** On ajoute également `a_row_context`, `b_row_context` et `row_context_radius`, soit un extrait des lignes voisines pour que les flux d’assistance consultent instantanément le contexte local.
- **[JA]** Markdown/Setext 見出しや `第○条` / `Section 4` などの条項タイトルも拾い、`section_heading` / `section_path` / `section_level`（+ `_a` / `_b`）をイベントと `assist_plan.json` に反映するため、git 風 diff でも「どの章節が変化したか」が一目瞭然です。
- **[EN]** Markdown + Setext titles and legal headings such as `第○条` or `Section 4` are captured and forwarded as `section_heading`, `section_path`, and `section_level` (with `_a` / `_b` variants) so the git-style quick diff and its assist plan always identify the chapter that changed.
- **[FR]** Les titres Markdown/Setext ainsi que les en-têtes juridiques (`第○条`, `Section 4`, etc.) sont détectés puis copiés dans les événements et `assist_plan.json` via `section_heading`, `section_path`, `section_level` (plus les variantes `_a` / `_b`), rendant la localisation des changements évidente même dans le diff façon git.
- **[JA]** 複数の数値を含む行では Hungarian 法ベースのペアリングを使い、通貨/単位/パーセントの不一致にペナルティを課したうえで最適な組を選びます。`--simple_pair_threshold` で許容コストを調整でき、イベントには `line_pair_cost` / `line_pair_gap` / `line_pair_penalty` / `line_pair_status` が入るため、どの値がマッチし残差かをひと目で把握できます。
- **[EN]** Lines carrying multiple numbers now run through a Hungarian-style assignment that favours the closest values while penalising currency/unit/percent mismatches; tune the acceptance window via `--simple_pair_threshold`. Each event exposes `line_pair_cost`, `line_pair_gap`, `line_pair_penalty`, and `line_pair_status` so you can tell which figures aligned and which ones remained unmatched.
- **[FR]** Les lignes contenant plusieurs montants passent désormais par un appariement de type hongrois qui privilégie les valeurs les plus proches et pénalise les divergences d’unité/devise/pourcentage ; la fenêtre d’acceptation se règle via `--simple_pair_threshold`. Les événements incluent `line_pair_cost`, `line_pair_gap`, `line_pair_penalty` et `line_pair_status`, ce qui précise quelles valeurs ont été couplées ou laissées sans correspondance.

```bash
python -m zocr.diff \
  --a out/A --b out/B \
  --simple_text_a memo_v1.txt \
  --simple_text_b memo_v2.txt \
  --simple_diff_out out/diff/memo.diff \
  --simple_json_out out/diff/memo.numeric.json \
  --simple_plan_out out/diff/memo.assist.json \
  --simple_markdown_out out/diff/memo.md
```

```bash
# semantic diff を省き、軽量モードのみ実行する例
# Example showing the quick differ on its own / Exemple en mode léger seul
python -m zocr.diff \
  --simple_text_a memo_v1.txt \
  --simple_text_b memo_v2.txt \
  --simple_diff_out out/diff/memo.diff \
  --simple_json_out out/diff/memo.numeric.json \
  --simple_plan_out out/diff/memo.assist.json \
  --simple_markdown_out out/diff/memo.md
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
