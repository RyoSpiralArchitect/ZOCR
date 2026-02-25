# Z-OCR Pilot Proposal (1‑slide) / Z-OCR パイロット提案（1枚）

**Client / クライアント:** (TBD)  
**Project / 案件:** (TBD)  
**Date / 作成日:** (TBD)  

This document is for scoping only. Final terms are provided in a separate agreement.  
本書は要件整理のための1枚提案です。最終条件は別途の商用契約で確定します（※法的助言ではありません）。

---

## What you get / 得られる価値

- **Doc understanding pipeline**: OCR → table reconstruction → export → index/RAG bundle → monitoring/tuning → report/diff
- **Fast internal deployment**: Docker / `docker compose` based reference API (`/healthz`, `/v1/run`, `/v1/run.zip`)
- **Auditability**: run artifacts (JSONL/SQL/report) + reproducibility metadata (`pipeline_meta.json`)

---

## Pilot scope (default) / パイロット範囲（標準）

- Deployment: **On‑prem (client-operated)** / オンプレ（クライアント運用）
- Environments: **1** (single sandbox) / 1環境
- Concurrency: **1** worker slot / 同時実行1
- Volume guideline: **~20,000 pages/month** / 月2万ページ目安
- Doc types: up to **2** categories to start (e.g., invoices + bank statements) / まず2種まで
- Outputs: JSONL + report (+ optional SQL/RAG bundle) / JSONL＋レポート（必要ならSQL/RAG）

Out of scope unless added / 別途オプション:
- Large-scale integrations (DWH, enterprise SSO), custom SLAs, heavy UI work

---

## Deliverables / 成果物

- Working pilot deployment (Docker/Compose) + runbook / 動作する導入一式＋運用メモ
- Sample processing flows for agreed doc types / 合意文書の処理フロー
- Findings report: quality, failure modes, tuning knobs, next-step recommendation / 品質・失敗要因・改善案
- Recommendation for production tier (Basic/Pro/Enterprise) / 本番ティア提案

---

## Success criteria (example) / 成功指標（例）

- Measurable improvements on agreed fields (amount/date/vendor/etc.) / 主要フィールド精度の改善
- Stable batch runs with documented parameters / 安定運用できるパラメータ確立
- Clear “go / no-go” decision with cost-to-production estimate / 本番移行の判断材料

---

## Timeline (typical) / 期間（目安）

- **Week 1**: kickoff + environment + sample docs + acceptance criteria / キックオフ＋環境＋受入条件
- **Week 2–4**: baseline runs + failure taxonomy + quick wins / ベースライン＋失敗分類＋改善
- **Week 5–8**: tuning + monitoring + regression checks / チューニング＋監視＋回帰
- **Week 9–12**: handoff + production sizing proposal / 引き継ぎ＋本番見積

---

## Commercial (Pilot) / 商用条件（パイロット）

- Price: **¥500,000 / 3 months** (excl. tax) / **50万円（税別）/3ヶ月**
- Support: **Basic** (see `SUPPORT.md`) / Basic（`SUPPORT.md`参照）
- Add-ons: PS **¥150,000/day** if needed / 必要に応じPS 15万円/日

Details: `PRICING.md`, `SUPPORT.md`  

---

## Next steps / 次アクション

1) Fill `sales/CLIENT_INTAKE.md` / ヒアリング記入  
2) Provide 20–50 representative docs + constraints (PII/region/retention) / 代表データ提供  
3) Confirm pilot acceptance criteria + timeline / 受入条件と日程合意  

