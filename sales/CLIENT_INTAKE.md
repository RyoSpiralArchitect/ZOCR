# Client Intake (1‑pager) / ヒアリングシート（1ページ）

This document is for scoping and quoting only. Final terms are provided in a separate agreement.  
本シートは要件整理と見積のためのものです。最終条件は別途の商用契約で定義します（※法的助言ではありません）。

---

## 1) Basic info / 基本情報

- Company / 会社名:
- Contact / 担当者:
- Email:
- Intended deployment start date / 導入希望時期:
- Decision deadline / 稟議・決裁期限:

---

## 2) Deployment type / 提供形態

Choose one / いずれか:
- [ ] On‑prem (client-operated) / オンプレ（クライアント運用）
- [ ] Managed service (private hosting) / マネージド（専用運用）

Environments needed / 必要環境:
- [ ] dev  [ ] stage  [ ] prod  (count: ___)

Concurrency (worker slots) / 同時実行枠:
- Desired / 希望: ___
- Max peak / 最大ピーク: ___

---

## 3) Data & compliance / データ・コンプライアンス

- Document types / 文書種別（例: 請求書、発注書、明細、契約…）:
- Languages / 言語（JA/EN/FR/…）:
- PII present? / 個人情報あり？:  [ ] yes  [ ] no  [ ] unknown
- Data region constraints / データ保管リージョン制約（例: JP only）:
- Retention / 保持期間（ログ・成果物）: ___ days
- Security requirements / セキュリティ要件（SAST/監査/暗号化など）:

---

## 4) Volume & performance / 規模・性能

**Primary billing unit is pages** (1 PDF page = 1 page; 1 image = 1 page).  
課金の基本単位はページ（PDF 1ページ=1ページ、画像1枚=1ページ）です。

- Monthly volume (range) / 月間ページ数（レンジ）:
  - [ ] <20k  [ ] 20k–100k  [ ] 100k–500k  [ ] 500k–2M  [ ] >2M
- Peak (pages/day) / ピーク（ページ/日）: ___
- Typical DPI / 想定DPI: [ ] 200  [ ] 300  [ ] other: ___
- Latency target / レイテンシ目標:
  - [ ] batch ok  [ ] interactive  [ ] strict SLA

---

## 5) Output contract / 出力要件（契約）

Pick the outputs you need / 必要な出力:
- [ ] JSONL (cells/sections)
- [ ] SQL export / SQL
- [ ] RAG bundle
- [ ] Diff reports (HTML/.diff/events)
- [ ] API only (FastAPI wrapper)

Integrations / 連携先:
- [ ] S3/MinIO  [ ] SharePoint  [ ] Box/Drive  [ ] DB  [ ] Search  [ ] other: ___

---

## 6) Support / サポート

Support tier / 希望サポート:
- [ ] Basic  [ ] Pro  [ ] Enterprise

SLA needed? / SLA必要？:
- [ ] yes (target: ___)  [ ] no  [ ] unsure

---

## 7) Pricing recommendation (internal) / ティア当て（社内用）

Use this as a quick guide; final sizing is by agreement.  
目安です（最終は別途合意）。

- Pilot: evaluation / non-prod
- Production Basic: up to ~100k pages/month, concurrency ~2, up to 2 env
- Production Pro: up to ~500k pages/month, concurrency ~4, 3 env, faster support
- Enterprise: SLA/region/compliance/high volume/custom integrations

---

## Notes / 備考

- Known constraints / 既知の制約:
- Risks / リスク:
- Next steps / 次アクション:

