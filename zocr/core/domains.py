"""Domain presets and keyword tables."""
from __future__ import annotations

from typing import Dict, List, Tuple

try:  # pragma: no cover - optional shared resources
    from zocr.resources.domain_dictionary import (
        all_domain_keywords as _core_all_domain_keywords,
    )  # type: ignore
except Exception:  # pragma: no cover - optional shared resources
    _core_all_domain_keywords = None  # type: ignore

__all__ = [
    "DOMAIN_DEFAULTS",
    "DOMAIN_KW",
    "DOMAIN_SUGGESTED_QUERIES",
    "DOMAIN_MONITOR_QUERIES",
    "DOMAIN_ALIAS",
    "DOMAIN_HEADER_SIGNALS",
    "HEADER_CONCEPT_SIGNALS",
]

# copy of the curated keyword list
_STATIC_DOMAIN_KW = {
    "invoice": [("合計",1.0),("金額",0.9),("消費税",0.8),("小計",0.6),("請求",0.4),("登録",0.3),("住所",0.3),("単価",0.3),("数量",0.3)],
    "invoice_jp_v2": [("合計",1.0),("金額",0.9),("消費税",0.8),("小計",0.6),("請求日",0.5),("発行日",0.4)],
    "invoice_en": [("invoice",1.0),("total",0.9),("amount",0.85),("tax",0.7),("due",0.5),("bill",0.4)],
    "invoice_fr": [("facture",1.0),("total",0.9),("montant",0.8),("tva",0.7),("échéance",0.5),("paiement",0.4)],
    "purchase_order": [("purchase",1.0),("order",0.95),("po",0.8),("qty",0.6),("ship",0.5),("vendor",0.4)],
    "expense": [("expense",1.0),("reimbursement",0.85),("category",0.6),("receipt",0.6),("total",0.5)],
    "timesheet": [("timesheet",1.0),("hours",0.95),("project",0.6),("rate",0.5),("overtime",0.4)],
    "shipping_notice": [("shipment",1.0),("tracking",0.9),("carrier",0.7),("delivery",0.6),("ship",0.5)],
    "medical_receipt": [("診療",1.0),("点数",0.9),("保険",0.8),("負担金",0.6),("薬剤",0.5)],
    "contract": [("契約",0.8),("署名",0.6),("印",0.5),("住所",0.3),("日付",0.3)],
    "contract_jp_v2": [("契約",0.9),("条",0.7),("締結",0.6),("甲",0.5),("乙",0.5),("印",0.4)],
    "contract_en": [("contract",1.0),("signature",0.75),("party",0.6),("term",0.6),("agreement",0.5)],
    "delivery": [("納品",1.0),("数量",0.8),("単位",0.5),("品名",0.5),("受領",0.4)],
    "delivery_jp": [("納品書",1.0),("数量",0.85),("品番",0.6),("受領",0.5),("出荷",0.4)],
    "delivery_en": [("delivery",1.0),("ship",0.85),("carrier",0.7),("qty",0.6),("item",0.5)],
    "estimate": [("見積",1.0),("単価",0.8),("小計",0.6),("有効期限",0.4)],
    "estimate_jp": [
        ("見積書", 4.0),
        ("御見積書", 3.6),
        ("見積日", 2.3),
        ("見積金額", 3.0),
        ("御見積金額", 3.0),
        ("御見積合計", 2.7),
        ("合計金額", 2.4),
        ("合計", 2.0),
        ("総計", 1.8),
        ("総額", 1.8),
        ("計", 1.5),
        ("小計", 1.6),
        ("数量", 1.2),
        ("単価", 1.2),
        ("金額", 1.2),
        ("有効期限", 2.5),
        ("お見積有効期限", 2.3),
        ("見積有効期限", 2.3),
        ("納期", 1.4),
        ("消費税", 1.5),
        ("税込", 1.2),
        ("税抜", 1.0),
    ],
    "estimate_en": [("estimate",1.0),("quote",0.9),("valid",0.6),("subtotal",0.6),("project",0.4)],
    "receipt": [("領収",1.0),("金額",0.9),("受領",0.6),("発行日",0.4),("住所",0.3)],
    "receipt_jp": [("領収書",1.0),("税込",0.8),("受領",0.6),("発行日",0.4)],
    "receipt_en": [("receipt",1.0),("paid",0.9),("total",0.75),("payment",0.6),("tax",0.5)],
    "bank_statement_en": [("statement",1.0),("account",0.9),("balance",0.8),("transaction",0.7),("debit",0.6),("credit",0.6),("bank",0.5)],
    "bank_statement_jp": [("取引明細",1.0),("口座番号",0.9),("残高",0.8),("入金",0.7),("出金",0.7),("金融機関",0.6),("支店",0.5)],
    "utility_bill_en": [("utility",1.0),("electric",0.85),("gas",0.8),("water",0.75),("kwh",0.6),("meter",0.55),("billing",0.5)],
    "utility_bill_jp": [("ご使用量",1.0),("電気",0.85),("ガス",0.8),("水道",0.75),("検針",0.6),("請求額",0.55),("契約",0.5)],
    "insurance_claim_en": [("claim",1.0),("policy",0.85),("insured",0.75),("coverage",0.65),("adjuster",0.55),("deductible",0.5)],
    "insurance_claim_jp": [("保険金請求",1.0),("被保険者",0.85),("保険証券",0.75),("事故日",0.65),("給付",0.55),("診断書",0.5)],
    "tax_form_en": [("tax",1.0),("return",0.9),("irs",0.8),("deduction",0.7),("withholding",0.6),("income",0.6)],
    "tax_form_jp": [("確定申告",1.0),("所得税",0.9),("控除",0.75),("課税",0.65),("源泉",0.6),("扶養",0.5)],
    "payslip_en": [("payslip",1.0),("payroll",0.9),("gross",0.75),("net pay",0.7),("deductions",0.65),("hours",0.55)],
    "payslip_jp": [("給与明細",1.0),("支給額",0.9),("控除",0.75),("差引支給額",0.7),("残業",0.6),("社会保険料",0.55)],
    "rental_agreement_en": [("rental",1.0),("lease",0.95),("tenant",0.75),("landlord",0.7),("premises",0.6)],
    "rental_agreement_jp": [("賃貸借",1.0),("賃料",0.9),("借主",0.75),("貸主",0.75),("物件",0.6),("契約期間",0.5)],
    "loan_statement_en": [("loan",1.0),("interest",0.9),("principal",0.85),("installment",0.7),("statement",0.6),("balance",0.6)],
    "loan_statement_jp": [("返済",1.0),("借入",0.9),("利息",0.85),("元金",0.75),("残高",0.65),("明細",0.55)],
    "travel_itinerary_en": [("itinerary",1.0),("flight",0.9),("departure",0.85),("arrival",0.85),("hotel",0.7),("booking",0.6)],
    "travel_itinerary_jp": [("旅程",1.0),("出発",0.9),("到着",0.9),("航空券",0.75),("宿泊",0.65),("予約",0.6)],
    "medical_bill_en": [("medical",1.0),("invoice",0.9),("patient",0.85),("procedure",0.7),("amount",0.6),("insurance",0.55)],
    "medical_bill_jp": [("診療",1.0),("請求",0.9),("患者",0.8),("保険",0.75),("点数",0.65),("金額",0.6)],
    "customs_declaration_en": [("customs",1.0),("declaration",0.95),("tariff",0.75),("shipment",0.7),("origin",0.6),("duty",0.55)],
    "grant_application_en": [("grant",1.0),("fund",0.9),("proposal",0.8),("budget",0.7),("milestone",0.6)],
    "boarding_pass_en": [("boarding",1.0),("flight",0.95),("seat",0.85),("gate",0.7),("departure",0.65),("passenger",0.6)],
}

DOMAIN_DEFAULTS = {
    "invoice": {"lambda_shape": 4.5, "w_kw": 0.6, "w_img": 0.3, "ocr_min_conf": 0.55},
    "invoice_jp_v2": {"lambda_shape": 4.5, "w_kw": 0.6, "w_img": 0.3, "ocr_min_conf": 0.55},
    "invoice_en": {"lambda_shape": 4.2, "w_kw": 0.55, "w_img": 0.25, "ocr_min_conf": 0.53},
    "invoice_fr": {"lambda_shape": 4.2, "w_kw": 0.55, "w_img": 0.25, "ocr_min_conf": 0.53},
    "purchase_order": {"lambda_shape": 4.0, "w_kw": 0.5, "w_img": 0.22, "ocr_min_conf": 0.60},
    "expense": {"lambda_shape": 3.8, "w_kw": 0.5, "w_img": 0.18, "ocr_min_conf": 0.60},
    "timesheet": {"lambda_shape": 3.6, "w_kw": 0.45, "w_img": 0.18, "ocr_min_conf": 0.62},
    "shipping_notice": {"lambda_shape": 4.3, "w_kw": 0.5, "w_img": 0.26, "ocr_min_conf": 0.58},
    "medical_receipt": {"lambda_shape": 5.0, "w_kw": 0.65, "w_img": 0.28, "ocr_min_conf": 0.60},
    "contract": {"lambda_shape": 4.0, "w_kw": 0.55, "w_img": 0.2, "ocr_min_conf": 0.60},
    "contract_jp_v2": {"lambda_shape": 4.1, "w_kw": 0.6, "w_img": 0.2, "ocr_min_conf": 0.60},
    "contract_en": {"lambda_shape": 4.0, "w_kw": 0.55, "w_img": 0.2, "ocr_min_conf": 0.60},
    "delivery": {"lambda_shape": 4.2, "w_kw": 0.5, "w_img": 0.25, "ocr_min_conf": 0.58},
    "delivery_jp": {"lambda_shape": 4.2, "w_kw": 0.5, "w_img": 0.25, "ocr_min_conf": 0.58},
    "delivery_en": {"lambda_shape": 4.0, "w_kw": 0.48, "w_img": 0.22, "ocr_min_conf": 0.58},
    "estimate": {"lambda_shape": 4.3, "w_kw": 0.55, "w_img": 0.25, "ocr_min_conf": 0.58},
    "estimate_jp": {"lambda_shape": 4.3, "w_kw": 0.55, "w_img": 0.25, "ocr_min_conf": 0.58},
    "estimate_en": {"lambda_shape": 4.2, "w_kw": 0.5, "w_img": 0.25, "ocr_min_conf": 0.58},
    "receipt": {"lambda_shape": 4.1, "w_kw": 0.6, "w_img": 0.2, "ocr_min_conf": 0.60},
    "receipt_jp": {"lambda_shape": 4.1, "w_kw": 0.6, "w_img": 0.2, "ocr_min_conf": 0.60},
    "receipt_en": {"lambda_shape": 4.0, "w_kw": 0.55, "w_img": 0.2, "ocr_min_conf": 0.60},
    "bank_statement_en": {"lambda_shape": 4.1, "w_kw": 0.58, "w_img": 0.24, "ocr_min_conf": 0.60},
    "bank_statement_jp": {"lambda_shape": 4.2, "w_kw": 0.6, "w_img": 0.24, "ocr_min_conf": 0.60},
    "utility_bill_en": {"lambda_shape": 3.9, "w_kw": 0.52, "w_img": 0.22, "ocr_min_conf": 0.58},
    "utility_bill_jp": {"lambda_shape": 4.0, "w_kw": 0.55, "w_img": 0.22, "ocr_min_conf": 0.58},
    "insurance_claim_en": {"lambda_shape": 4.4, "w_kw": 0.6, "w_img": 0.24, "ocr_min_conf": 0.60},
    "insurance_claim_jp": {"lambda_shape": 4.5, "w_kw": 0.62, "w_img": 0.24, "ocr_min_conf": 0.60},
    "tax_form_en": {"lambda_shape": 4.6, "w_kw": 0.63, "w_img": 0.22, "ocr_min_conf": 0.60},
    "tax_form_jp": {"lambda_shape": 4.6, "w_kw": 0.65, "w_img": 0.22, "ocr_min_conf": 0.60},
    "payslip_en": {"lambda_shape": 3.9, "w_kw": 0.58, "w_img": 0.2, "ocr_min_conf": 0.62},
    "payslip_jp": {"lambda_shape": 4.0, "w_kw": 0.6, "w_img": 0.2, "ocr_min_conf": 0.62},
    "rental_agreement_en": {"lambda_shape": 4.2, "w_kw": 0.6, "w_img": 0.22, "ocr_min_conf": 0.60},
    "rental_agreement_jp": {"lambda_shape": 4.2, "w_kw": 0.63, "w_img": 0.22, "ocr_min_conf": 0.60},
    "loan_statement_en": {"lambda_shape": 4.1, "w_kw": 0.6, "w_img": 0.24, "ocr_min_conf": 0.60},
    "loan_statement_jp": {"lambda_shape": 4.2, "w_kw": 0.62, "w_img": 0.24, "ocr_min_conf": 0.60},
    "travel_itinerary_en": {"lambda_shape": 3.8, "w_kw": 0.55, "w_img": 0.26, "ocr_min_conf": 0.58},
    "travel_itinerary_jp": {"lambda_shape": 3.9, "w_kw": 0.58, "w_img": 0.26, "ocr_min_conf": 0.58},
    "medical_bill_en": {"lambda_shape": 4.8, "w_kw": 0.64, "w_img": 0.24, "ocr_min_conf": 0.60},
    "medical_bill_jp": {"lambda_shape": 4.9, "w_kw": 0.66, "w_img": 0.24, "ocr_min_conf": 0.60},
    "customs_declaration_en": {"lambda_shape": 4.2, "w_kw": 0.58, "w_img": 0.22, "ocr_min_conf": 0.58},
    "grant_application_en": {"lambda_shape": 4.0, "w_kw": 0.6, "w_img": 0.22, "ocr_min_conf": 0.58},
    "boarding_pass_en": {"lambda_shape": 3.5, "w_kw": 0.5, "w_img": 0.3, "ocr_min_conf": 0.55},
}

for _dom_conf in DOMAIN_DEFAULTS.values():
    _dom_conf.setdefault("w_sym", 0.45)

DOMAIN_ALIAS = {
    "invoice": "invoice_jp_v2",
    "invoice_ja": "invoice_jp_v2",
    "invoice_en": "invoice_en",
    "contract": "contract_jp_v2",
    "delivery": "delivery_jp",
    "estimate": "estimate_jp",
    "receipt": "receipt_jp",
    "bank_statement": "bank_statement_en",
    "bank_statement_ja": "bank_statement_jp",
    "bank_statement_en": "bank_statement_en",
    "utility_bill": "utility_bill_en",
    "utility_bill_ja": "utility_bill_jp",
    "utility_bill_en": "utility_bill_en",
    "insurance_claim": "insurance_claim_en",
    "insurance_claim_ja": "insurance_claim_jp",
    "insurance_claim_en": "insurance_claim_en",
    "tax_form": "tax_form_en",
    "tax_form_ja": "tax_form_jp",
    "tax_form_en": "tax_form_en",
    "tax_return": "tax_form_en",
    "payslip": "payslip_en",
    "payslip_ja": "payslip_jp",
    "payslip_en": "payslip_en",
    "rental_agreement": "rental_agreement_en",
    "rental_agreement_ja": "rental_agreement_jp",
    "rental_agreement_en": "rental_agreement_en",
    "lease_contract": "rental_agreement_en",
    "loan_statement": "loan_statement_en",
    "loan_statement_ja": "loan_statement_jp",
    "loan_statement_en": "loan_statement_en",
    "loan_summary": "loan_statement_en",
    "travel_itinerary": "travel_itinerary_en",
    "travel_itinerary_ja": "travel_itinerary_jp",
    "travel_itinerary_en": "travel_itinerary_en",
    "travel_plan": "travel_itinerary_en",
    "medical_bill": "medical_bill_en",
    "medical_bill_ja": "medical_bill_jp",
    "medical_bill_en": "medical_bill_en",
    "medical_invoice": "medical_bill_en",
    "customs_declaration": "customs_declaration_en",
    "customs_declaration_ja": "customs_declaration_en",
    "customs_declaration_en": "customs_declaration_en",
    "customs_form": "customs_declaration_en",
    "grant_application": "grant_application_en",
    "grant_application_ja": "grant_application_en",
    "grant_application_en": "grant_application_en",
    "boarding_pass": "boarding_pass_en",
    "boarding_pass_ja": "boarding_pass_en",
    "boarding_pass_en": "boarding_pass_en",
    "purchase_order_ja": "purchase_order",
    "purchase_order_en": "purchase_order",
    "shipping_notice_ja": "shipping_notice",
    "shipping_notice_en": "shipping_notice",
    "expense_report_ja": "expense",
    "expense_report_en": "expense",
}

def _build_domain_kw() -> Dict[str, List[Tuple[str, float]]]:
    base: Dict[str, List[Tuple[str, float]]] = {
        dom: list(entries) for dom, entries in _STATIC_DOMAIN_KW.items()
    }
    if _core_all_domain_keywords is None:
        return base
    try:
        mapping = _core_all_domain_keywords()
    except Exception:
        return base
    for token, words in mapping.items():
        if not words:
            continue
        target = token if token in base else DOMAIN_ALIAS.get(token)
        if not target or target not in base:
            continue
        seen = {kw for kw, _ in base[target]}
        for word in sorted(words):
            if word in seen:
                continue
            weight = 0.45 if len(word) <= 3 else 0.35 if len(word) <= 6 else 0.25
            base[target].append((word, weight))
            seen.add(word)
    return base


DOMAIN_KW = _build_domain_kw()
DOMAIN_HEADER_SIGNALS: Dict[str, List[Tuple[str, float]]] = {
    "invoice": [("qty", 0.35), ("unit price", 0.35), ("amount", 0.45)],
    "invoice_jp_v2": [("数量", 0.35), ("単価", 0.35), ("金額", 0.45)],
    "invoice_en": [("qty", 0.3), ("unit", 0.3), ("amount", 0.4)],
    "purchase_order": [("po", 0.4), ("ship", 0.3), ("vendor", 0.3)],
    "delivery": [("qty", 0.3), ("unit", 0.3), ("delivery", 0.35)],
    "estimate": [("qty", 0.3), ("unit", 0.3), ("subtotal", 0.35)],
    "receipt": [("received", 0.35), ("payment", 0.3), ("total", 0.35)],
}
HEADER_CONCEPT_SIGNALS: Dict[str, List[Tuple[str, float]]] = {
    "table": [("invoice", 0.3), ("receipt", 0.3), ("estimate", 0.3)],
    "amount": [("invoice", 0.4), ("estimate", 0.3), ("receipt", 0.3)],
    "date": [("invoice", 0.2), ("contract", 0.3)],
    "ship": [("purchase_order", 0.4), ("delivery", 0.4)],
}

DOMAIN_SUGGESTED_QUERIES = {
    "invoice_jp_v2": ["合計 金額", "消費税", "支払", "振込", "請求日"],
    "invoice_en": ["invoice total", "amount due", "tax", "payment due"],
    "contract_jp_v2": ["契約日", "契約金額", "甲乙"],
    "delivery_jp": ["納品", "受領", "数量"],
    "estimate_jp": ["御見積", "有効期限", "数量", "小計"],
    "receipt_jp": ["領収", "受領", "税込"],
    "default": ["total amount", "date", "company", "tax"],
}
DOMAIN_MONITOR_QUERIES = {
    "invoice_jp_v2": {
        "q_amount": ["合計", "金額", "請求金額", "振込金額"],
        "q_date": ["請求日", "発行日"],
        "q_due": ["支払期限", "振込期限"],
    },
    "invoice_en": {
        "q_amount": ["total", "amount due", "balance", "invoice total"],
        "q_date": ["invoice date", "issue date"],
        "q_due": ["due date", "payment due"],
    },
    "default": {
        "q_amount": ["total", "amount", "balance"],
        "q_date": ["date", "issued", "invoice date"],
        "q_due": ["due date", "payment due"],
    },
}
