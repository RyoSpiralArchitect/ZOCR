"""Built-in tesslite resources for the toy OCR pipeline."""
from __future__ import annotations

DEFAULT_SIGNATURE = "tesslite_builtin_v1"

# Core glyph inventory: ASCII digits/letters plus Japanese finance terms.
DEFAULT_GLYPHS = list(
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "¥$€.,-/%()[]{}#&+"
    "数量単価金額合計小計税込税抜御見積御請求納期期限日曜曜月火水木金土有効税率税額請求書見積書" 
)

DEFAULT_AMBIGUOUS = {
    "O": {"0"},
    "0": {"O"},
    "I": {"1", "l"},
    "l": {"1"},
    "１": {"1"},
    "５": {"5"},
    "Ｓ": {"S"},
    "￥": {"¥"},
}

DEFAULT_CHAR_CATEGORIES = {}
for digit in "0123456789":
    DEFAULT_CHAR_CATEGORIES[digit] = "digit"
for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz":
    DEFAULT_CHAR_CATEGORIES[letter] = "latin"
for symbol in "¥$€":
    DEFAULT_CHAR_CATEGORIES[symbol] = "currency"
for token in "数量単価金額合計小計税込税抜御見積御請求納期期限日税率税額請求書見積書":
    DEFAULT_CHAR_CATEGORIES[token] = "kanji"

DEFAULT_DICTIONARY = {
    "item",
    "items",
    "qty",
    "quantity",
    "unit",
    "unitprice",
    "unit price",
    "price",
    "amount",
    "total",
    "subtotal",
    "tax",
    "taxable",
    "tax rate",
    "tax amount",
    "balance",
    "due",
    "due date",
    "見積",
    "見積書",
    "御見積",
    "御見積金額",
    "見積金額",
    "数量",
    "単価",
    "金額",
    "小計",
    "合計",
    "合計金額",
    "税込",
    "税抜",
    "消費税",
    "税率",
    "納期",
    "有効期限",
    "請求金額",
    "御請求金額",
}

DEFAULT_BIGRAMS = {}
