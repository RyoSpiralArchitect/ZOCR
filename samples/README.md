# Samples / サンプル / Exemples

`samples/demo_inputs/` に PDF や PNG を配置すると、`--input demo` でこれらの実ファイルをデモ素材として処理します。
Drop your PDFs/PNGs into `samples/demo_inputs/` to drive `--input demo` with your own material.
Déposez vos PDF/PNG dans `samples/demo_inputs/` afin que `--input demo` s'appuie sur vos fichiers.

## 📂 One folder, many domains

サンプルはすべて `samples/demo_inputs/`
配下にまとめてください。ファイル名や CLI の `--domain` でコンテキストを切り替えられます。
All domain-specific subfolders have been merged into `samples/demo_inputs/`. Keep every sample there—the orchestrator and future
GUI will decide which domain profile to use.
Tous les exemples résident désormais dans `samples/demo_inputs/`; le domaine se choisit via `--domain` ou l'interface graphique.

## 🧭 Domain quick reference / ドメイン早見表

| Sample type / サンプル種別 | Suggested `--domain` | JA keywords (抜粋) | EN keywords (sample) |
| --- | --- | --- | --- |
| 請求書 / Invoice | `invoice`, `invoice_en` | 請求書, 請求日, 合計金額, 消費税, 支払期日 | invoice, total amount, tax, due date, billing address |
| 発注書 / Purchase order | `purchase_order` | 発注書, 発注番号, 納期, 仕入先, 品番, 数量, 単価 | purchase order, PO number, vendor, ship to, unit price, line item |
| 医療請求 / Medical bill | `medical_bill`, `medical_bill_en` | 診療明細, 保険, 患者氏名, 点数, 自己負担, 投薬 | medical bill, patient, provider, diagnosis, copay, procedure |
| 通関申告 / Customs declaration | `customs_declaration` | 通関, HSコード, 仕向地, 原産国, 課税価格 | customs declaration, tariff, origin, importer, duties |
| 助成金申請 / Grant application | `grant_application` | 助成金, 交付申請, 事業計画, 予算, 研究代表者 | grant application, funding amount, proposal, reviewer, milestone |
| 搭乗券 / Boarding pass | `boarding_pass` | 搭乗券, 便名, 出発時刻, 搭乗口, 座席番号 | boarding pass, flight, gate, boarding time, seat |
| 賃貸借契約 / Rental agreement | `rental_agreement` | 賃貸借契約書, 契約期間, 賃料, 敷金, 管理費, 物件住所 | lease agreement, rent, deposit, tenant, landlord |
| ローン明細 / Loan statement | `loan_statement` | 返済予定表, 元金, 利息, 返済額, 支払期日 | loan statement, principal, interest, installment, maturity |
| 旅行行程 / Travel itinerary | `travel_itinerary` | 旅程, 出発地, 到着地, 宿泊, 予約番号, 便名 | itinerary, departure, arrival, hotel, confirmation, booking reference |
| 銀行明細 / Bank statement | `bank_statement` | 取引明細, 口座番号, 振込, 入金, 引落, 残高 | bank statement, account number, deposit, withdrawal, balance |
| 公共料金 / Utility bill | `utility_bill` | 請求内訳, ご使用量, 検針日, 契約種別, 支払期限 | utility bill, usage, meter reading, billing period, due date |
| 保険金請求 / Insurance claim | `insurance_claim` | 保険金請求書, 被保険者, 事故日, 診断書, 給付額 | insurance claim, policy number, incident date, adjuster, payout |
| 税務申告 / Tax form | `tax_form` | 確定申告, 課税所得, 控除額, 源泉徴収, 申告区分 | tax form, taxable income, deduction, withholding, refund |
| 給与明細 / Payslip | `payslip` | 給与明細, 支給額, 控除, 差引支給額, 勤怠, 残業 | payslip, gross pay, net pay, deduction, overtime |
| 出荷案内 / Shipping notice | `shipping_notice` | 出荷案内, 納品書, 出荷日, 配送業者, 追跡番号 | shipping notice, shipment, tracking, carrier, ship date |
| 経費精算 / Expense report | `expense_report` | 経費精算書, 申請日, 立替, 交通費, 領収書, 承認者 | expense report, reimbursement, category, receipt, approver |

> ℹ️ キーワードは `zocr.resources.domain_dictionary` にも収録され、Toy OCR／consensus exporter／`zocr.core` の組込み辞書として利用されます。
The keyword lists above feed into `zocr.resources.domain_dictionary`, which now drives the bundled toy OCR lexicon, the consensus exporter, and the `zocr.core` boosts—no external wordlists are required.
Les listes de mots-clés ci-dessus alimentent `zocr.resources.domain_dictionary`, utilisé par le Toy OCR, l’exportateur consensus et les boosts `zocr.core`, sans dictionnaires externes.

`--domain`（またはパイプライン側の domain 設定）を指定すると、そのドメインのキーワード辞書が toy OCR の lexicon に適用されます。
Passing `--domain` (or configuring the pipeline domain) forces the toy OCR lexicon to load that domain's keyword bundle.
