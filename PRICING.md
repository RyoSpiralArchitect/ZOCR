# Pricing (Draft) / 料金（ドラフト）

This document is a **non-binding draft** intended to make commercial discussions easier.
Final terms are provided in a separate commercial agreement. This is **not legal advice**.

---

## Definitions / 用語

- **Page / ページ**: 1つの入力ページ（PDFは1ページ=1ページ、画像は1枚=1ページ）。通常の運用は `--dpi 200` を想定。
- **Environment / 環境**: 独立して運用されるデプロイ単位（例: dev / stage / prod）。
- **Concurrency / 同時実行**: 同時に走らせるパイプライン実行枠（ワーカースロット）。
- **Business day / 営業日**: 原則、平日（JST）で運用。詳細は契約で調整。

---

## Commercial license (On‑prem / internal) / 商用ライセンス（オンプレ・社内利用）

AGPL の要件に合わない使い方（例: 社内サービスとして提供しつつ改変を非公開にしたい、保証/SLAが必要など）の場合に利用します。

**Scope (default) / 想定スコープ（標準）**
- Single legal entity / 単一法人
- Internal use / 社内利用（第三者への再配布は別途合意が必要）
- Updates during term / 契約期間中のアップデート提供
- Client-operated infra / インフラはクライアント側で運用（オンプレ）

### Tiers (JPY, excl. tax) / ティア（税別・円）

| Tier | Intended use / 用途 | Included (baseline) / 含まれる範囲（目安） | Price |
|---|---|---|---:|
| PoC / Pilot | Evaluation / non-production | 1 env, concurrency 1, ~20k pages/month, Basic support | ¥500,000 / 3 months |
| Production Basic | Small internal production | 2 env (prod + non-prod), concurrency 2, ~100k pages/month, Basic support | ¥3,000,000 / year |
| Production Pro | Multiple teams / higher volume | 3 env (dev + stage + prod), concurrency 4, ~500k pages/month, Pro support | ¥6,000,000 / year |
| Enterprise | SLA / custom needs | Custom (env / concurrency / volume / SLA) | From ¥10,000,000 / year |

**Notes / 補足**
- The volume/concurrency baselines are for sizing and support planning (not a hard technical limit). If usage consistently exceeds the baseline, we’ll propose an upgrade or an addendum.  
  （ページ数/同時実行の目安は主に見積・サポート設計のため。恒常的に超える場合はアップグレードや追加契約を提案します）
- Pilot fee can be credited against the first-year license if you proceed to Production within 3 months (optional).  
  （Pilot費用を本番契約に充当するオプションは個別協議）

### Add-ons (examples) / 追加オプション（例）

- Extra environment: from **¥600,000 / year** per env
- Extra concurrency: from **¥300,000 / year** per +1 slot
- Priority security response / SLA uplift: quoted
- Professional services (integration, tuning, exporters): **¥150,000 / day**

Support options: `SUPPORT.md`

---

## Managed service (SaaS / private hosting) / マネージド提供（SaaS/専用運用）

運用（監視/更新/バックアップ等）込みでこちらが提供するプランです。月額＋従量（超過）を基本にします。

### Tiers (JPY, excl. tax) / ティア（税別・円）

| Tier | Included volume (per month) | Included (baseline) | Price |
|---|---:|---|---:|
| Starter | 20,000 pages | concurrency 1, retention 30 days, Basic support | ¥200,000 / month |
| Growth | 80,000 pages | concurrency 2, retention 90 days, Pro support | ¥500,000 / month |
| Enterprise | Custom | Custom | Custom |

**Overage / 超過**
- From **¥5 / page** (billed monthly). Price depends on OCR engine choices, doc complexity, and latency targets.

---

## Contact / 連絡先

For quotes, include / 見積に必要な情報:
- Deployment type: on‑prem / managed hosting
- Expected volume (pages/day or pages/month) and peak load
- Environments needed (dev/stage/prod)
- Data sensitivity constraints (PII, retention, regions)
- Support/SLA requirements

Contact: kishkavsesvit@icloud.com
