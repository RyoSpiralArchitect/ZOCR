"""Built-in tesslite resources for the toy OCR pipeline."""
from __future__ import annotations

from .domain_dictionary import ALL_KEYWORDS

DEFAULT_SIGNATURE = "tesslite_builtin_v2"

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

# Merge every domain dictionary into the built-in wordlist.
DEFAULT_DICTIONARY = sorted(ALL_KEYWORDS)

DEFAULT_BIGRAMS = {}
