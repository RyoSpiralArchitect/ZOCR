# -*- coding: utf-8 -*-
"""Lightweight text differ for ToyOCR-style comparisons.

This module keeps a "git diff" feel while highlighting numeric deltas so
similar-looking business documents (estimates, invoices, memos) can be
compared without loading the full semantic diff stack.
"""
from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _clip(text: str, limit: int = 200) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _label_name(label: str) -> str:
    try:
        return Path(label).name or label
    except Exception:
        return label


_CURRENCY_SYMBOLS = {
    "$": "USD",
    "€": "EUR",
    "¥": "JPY",
    "￥": "JPY",
    "£": "GBP",
    "₤": "GBP",
    "₩": "KRW",
    "₽": "RUB",
    "₹": "INR",
    "₨": "INR",
    "₺": "TRY",
    "₫": "VND",
    "₱": "PHP",
    "₦": "NGN",
    "₲": "PYG",
    "₴": "UAH",
    "₪": "ILS",
    "₡": "CRC",
    "₭": "LAK",
    "฿": "THB",
}

_CURRENCY_WORD_PREFIXES = {
    "USD": "USD",
    "US$": "USD",
    "EUR": "EUR",
    "JPY": "JPY",
    "CNY": "CNY",
    "RMB": "CNY",
    "HKD": "HKD",
    "TWD": "TWD",
    "NTD": "TWD",
    "NT$": "TWD",
    "CAD": "CAD",
    "AUD": "AUD",
    "NZD": "NZD",
    "SGD": "SGD",
    "GBP": "GBP",
    "CHF": "CHF",
    "SEK": "SEK",
    "NOK": "NOK",
    "DKK": "DKK",
    "KRW": "KRW",
    "IDR": "IDR",
    "INR": "INR",
    "MYR": "MYR",
    "MXN": "MXN",
    "BRL": "BRL",
    "ZAR": "ZAR",
    "AED": "AED",
    "SAR": "SAR",
    "TRY": "TRY",
    "ILS": "ILS",
    "VND": "VND",
    "PHP": "PHP",
    "THB": "THB",
    "COP": "COP",
    "ARS": "ARS",
}

_CURRENCY_SUFFIXES = {
    "円": "JPY",
    "円(税込)": "JPY",
    "円(税別)": "JPY",
    "円（税別）": "JPY",
    "円（税込）": "JPY",
    "ドル": "USD",
    "usd": "USD",
    "eur": "EUR",
    "jpy": "JPY",
    "cny": "CNY",
    "rmb": "CNY",
    "hkd": "HKD",
    "twd": "TWD",
    "ntd": "TWD",
    "cad": "CAD",
    "aud": "AUD",
    "gbp": "GBP",
    "chf": "CHF",
    "sgd": "SGD",
    "krw": "KRW",
    "myr": "MYR",
    "idr": "IDR",
    "thb": "THB",
    "vnd": "VND",
    "php": "PHP",
    "brl": "BRL",
    "mxn": "MXN",
    "zar": "ZAR",
    "aed": "AED",
    "sar": "SAR",
}

_SCALE_SUFFIXES: List[Tuple[str, float, bool]] = [
    ("億", 1e8, False),
    ("万", 1e4, False),
    ("千", 1e3, False),
    ("bn", 1e9, True),
    ("mm", 1e6, True),
    ("m", 1e6, True),
    ("b", 1e9, True),
    ("k", 1e3, True),
]

_NUM_RE = re.compile(
    r"""
    (?<![\w])
    \(?
    (?:[$€¥£₹₩₽₺₪₫₱฿₦₲₴]|USD|EUR|JPY|CNY|RMB|CAD|AUD|NZD|SGD|HKD|TWD|NTD|CHF|KRW|GBP|INR|IDR|MYR|PHP|THB|VND|BRL|MXN|ZAR|AED|SAR|SEK|NOK|DKK)?
    [\s]*
    [-+]?
    \d[\d,]*(?:\.\d+)?
    (?:\s*(?:k|m|b|bn|mm|千|万|億))?
    \)?
    (?:\s*(?:%|％))?
    (?:\s*(?:円|ドル|元|usd|eur|jpy|cny|rmb|cad|aud|gbp|chf|sgd|krw|hkd|twd|ntd)(?:\s*\([^)]*\))?)?
    """,
    re.IGNORECASE | re.VERBOSE,
)


class SimpleTextDiffer:
    """Compare two plain-text documents with git-like output."""

    def __init__(
        self,
        context_lines: int = 3,
        line_match_threshold: float = 0.6,
        number_pair_threshold: float = 1.25,
    ) -> None:
        self.context_lines = max(0, int(context_lines))
        self.line_match_threshold = float(line_match_threshold)
        # Maximum acceptable pairing cost (difference + penalties) when matching numbers.
        self.number_pair_threshold = float(number_pair_threshold)

    def compare_files(self, path_a: Path, path_b: Path) -> Dict[str, Any]:
        """Run the diff on two files."""

        lines_a = self._read_lines(path_a)
        lines_b = self._read_lines(path_b)
        return self.compare_lines(lines_a, lines_b, str(path_a), str(path_b))

    def compare_lines(
        self,
        lines_a: Sequence[str],
        lines_b: Sequence[str],
        label_a: str = "A",
        label_b: str = "B",
    ) -> Dict[str, Any]:
        matcher = difflib.SequenceMatcher(a=list(lines_a), b=list(lines_b))
        opcodes = list(matcher.get_opcodes())
        diff_iter = difflib.unified_diff(
            lines_a,
            lines_b,
            fromfile=label_a,
            tofile=label_b,
            lineterm="",
            n=self.context_lines,
        )
        diff_text = "\n".join(diff_iter)

        numeric_changes = self._collect_numeric_changes(opcodes, lines_a, lines_b)
        summary = {
            "total_lines_a": len(lines_a),
            "total_lines_b": len(lines_b),
            "diff_hunks": sum(1 for tag, *_ in opcodes if tag != "equal"),
            "diff_lines": sum(
                (i2 - i1) + (j2 - j1)
                for tag, i1, i2, j1, j2 in opcodes
                if tag != "equal"
            ),
            "numeric_changes": len(numeric_changes),
            "line_match_threshold": self.line_match_threshold,
        }
        return {
            "diff": diff_text,
            "numeric_changes": numeric_changes,
            "summary": summary,
            "labels": {"a": label_a, "b": label_b},
        }

    # ------------------------------------------------------------------
    def events_from_result(
        self,
        result: Dict[str, Any],
        label_a: str,
        label_b: str,
    ) -> List[Dict[str, Any]]:
        """Convert numeric changes into assist-plan friendly events."""

        events: List[Dict[str, Any]] = []
        doc_a = _label_name(label_a)
        doc_b = _label_name(label_b)
        table_id = f"simple_text::{doc_a}::{doc_b}"
        headers = [doc_a, doc_b]
        summary = result.get("summary", {})
        total_rows = max(
            int(summary.get("total_lines_a") or 0),
            int(summary.get("total_lines_b") or 0),
        )
        for idx, change in enumerate(result.get("numeric_changes", []), start=1):
            line_a = change.get("line_a")
            line_b = change.get("line_b")
            row_key = f"line_{line_a or 'na'}_vs_{line_b or 'na'}"
            a_context = self._context_preview(change.get("context_a"))
            b_context = self._context_preview(change.get("context_b"))
            context_radius = change.get("context_radius") or self._context_radius_for_metadata()
            base_event = {
                "type": "cell_updated",
                "source": "simple_text_differ",
                "table_id": table_id,
                "table_page": None,
                "table_index": 0,
                "table_rows": total_rows or None,
                "table_columns": 2,
                "table_headers": headers,
                "table_header_preview": f"{doc_a} ↔ {doc_b}",
                "row_key": row_key,
                "row_key_a": f"line_{line_a}" if line_a is not None else None,
                "row_key_b": f"line_{line_b}" if line_b is not None else None,
                "row_ids": [
                    f"{doc_a}:L{line_a}" if line_a is not None else None,
                    f"{doc_b}:L{line_b}" if line_b is not None else None,
                ],
                "a_row_preview": _clip(change.get("text_a", "")),
                "b_row_preview": _clip(change.get("text_b", "")),
                "a_row_context": a_context,
                "b_row_context": b_context,
                "row_context_radius": context_radius,
                "trace_a": f"{label_a}#L{line_a}" if line_a is not None else None,
                "trace_b": f"{label_b}#L{line_b}" if line_b is not None else None,
                "line_signature": change.get("line_signature"),
                "line_label": change.get("line_label"),
                "line_similarity": change.get("line_similarity"),
            }
            base_event["row_ids"] = [rid for rid in base_event["row_ids"] if rid]
            pairs = change.get("pairs") or []
            if not pairs:
                pairs = [
                    {
                        "old": None,
                        "new": None,
                        "delta": None,
                        "relative": None,
                        "old_raw": None,
                        "new_raw": None,
                    }
                ]
            for pair_idx, pair in enumerate(pairs, start=1):
                event = dict(base_event)
                event["row_key"] = f"{row_key}#{pair_idx}" if len(pairs) > 1 else row_key
                event["old"] = pair.get("old_raw") if pair.get("old_raw") is not None else pair.get("old")
                event["new"] = pair.get("new_raw") if pair.get("new_raw") is not None else pair.get("new")
                event["numeric_delta"] = pair.get("delta")
                event["relative_delta"] = pair.get("relative")
                event["similarity"] = None
                if pair.get("unit"):
                    event["numeric_unit"] = pair.get("unit")
                if pair.get("currency"):
                    event["numeric_currency"] = pair.get("currency")
                if pair.get("is_percent"):
                    event["numeric_is_percent"] = True
                if pair.get("scale"):
                    event["numeric_scale"] = pair.get("scale")
                if pair.get("pair_cost") is not None:
                    event["line_pair_cost"] = pair.get("pair_cost")
                if pair.get("pair_penalty") is not None:
                    event["line_pair_penalty"] = pair.get("pair_penalty")
                if pair.get("pair_gap") is not None:
                    event["line_pair_gap"] = pair.get("pair_gap")
                if pair.get("pair_status"):
                    event["line_pair_status"] = pair.get("pair_status")
                events.append(event)
        return events

    def _collect_numeric_changes(
        self,
        opcodes,
        lines_a: Sequence[str],
        lines_b: Sequence[str],
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        context_radius = self._context_radius_for_metadata()
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == "equal":
                continue
            slice_a = list(enumerate(lines_a[i1:i2], start=i1 + 1))
            slice_b = list(enumerate(lines_b[j1:j2], start=j1 + 1))
            aligned = self._align_slices(slice_a, slice_b)
            for left, right, score in aligned:
                if left is None and right is None:
                    continue
                line_no_a, text_a = left if left else (None, "")
                line_no_b, text_b = right if right else (None, "")

                nums_a = self._extract_numbers(text_a)
                nums_b = self._extract_numbers(text_b)
                if not nums_a and not nums_b:
                    continue
                pairs = self._pair_numbers(nums_a, nums_b)
                signature = self._line_signature(text_a or text_b)
                label = self._line_label(text_a or text_b)
                results.append(
                    {
                        "line_a": line_no_a,
                        "line_b": line_no_b,
                        "text_a": text_a,
                        "text_b": text_b,
                        "numbers_a": [token["raw"] for token in nums_a],
                        "numbers_b": [token["raw"] for token in nums_b],
                        "pairs": pairs,
                        "line_signature": signature,
                        "line_label": label,
                        "line_similarity": score,
                        "context_radius": context_radius,
                        "context_a": self._gather_context(lines_a, line_no_a, context_radius),
                        "context_b": self._gather_context(lines_b, line_no_b, context_radius),
                    }
                )
        return results

    def _align_slices(
        self,
        slice_a: Sequence,
        slice_b: Sequence,
    ) -> List[tuple]:
        """Pair lines by fuzzy text similarity (numbers blanked out)."""

        if not slice_a and not slice_b:
            return []

        remaining_b = list(range(len(slice_b)))
        aligned: List[tuple] = []
        for left in slice_a:
            best_idx = None
            best_score = 0.0
            left_sig = self._matchable_text(left[1])
            for b_idx in remaining_b:
                right = slice_b[b_idx]
                right_sig = self._matchable_text(right[1])
                if not left_sig and not right_sig:
                    score = 1.0
                else:
                    score = difflib.SequenceMatcher(a=left_sig, b=right_sig).ratio()
                if score > best_score:
                    best_idx = b_idx
                    best_score = score
            if best_idx is not None and best_score >= self.line_match_threshold:
                aligned.append((left, slice_b[best_idx], best_score))
                remaining_b.remove(best_idx)
            else:
                aligned.append((left, None, 0.0))

        for b_idx in remaining_b:
            aligned.append((None, slice_b[b_idx], 0.0))

        return aligned

    @staticmethod
    def _matchable_text(text: str) -> str:
        if not text:
            return ""
        lowered = text.lower()
        cleaned = _NUM_RE.sub("<num>", lowered)
        return re.sub(r"\s+", " ", cleaned).strip()

    def _line_signature(self, text: str) -> Optional[str]:
        signature = self._matchable_text(text)
        return signature or None

    def _line_label(self, text: str) -> Optional[str]:
        signature = self._matchable_text(text)
        if not signature:
            return None
        label = signature.replace("<num>", "").strip(" :|-•")
        return label or None

    @staticmethod
    def _extract_numbers(text: str) -> List[Dict[str, Any]]:
        tokens: List[Dict[str, Any]] = []
        if not text:
            return tokens
        scan_text = text.replace("（", "(").replace("）", ")")
        for match in _NUM_RE.finditer(scan_text):
            raw = match.group(0)
            parsed = SimpleTextDiffer._normalize_numeric_token(raw)
            if parsed:
                tokens.append(parsed)
        return tokens

    def _pair_numbers(
        self,
        nums_a: Sequence[Dict[str, Any]],
        nums_b: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        len_a, len_b = len(nums_a), len(nums_b)
        if not len_a and not len_b:
            return []
        if not len_a or not len_b:
            tokens = nums_a or nums_b
            pairs: List[Dict[str, Any]] = []
            for token in tokens:
                pairs.append(self._single_side_pair(token if nums_a else None, token if nums_b else None))
            return pairs

        size = max(len_a, len_b)
        high_cost = self.number_pair_threshold + 1.0
        cost_matrix = [[high_cost] * size for _ in range(size)]
        penalty_matrix = [[0.0] * size for _ in range(size)]
        gap_matrix = [[0.0] * size for _ in range(size)]
        for i in range(len_a):
            for j in range(len_b):
                total_cost, penalty, gap = self._numeric_pair_cost(nums_a[i], nums_b[j])
                cost_matrix[i][j] = total_cost
                penalty_matrix[i][j] = penalty
                gap_matrix[i][j] = gap

        assignment = self._hungarian(cost_matrix)
        matched_a = set()
        matched_b = set()
        pairs: List[Dict[str, Any]] = []
        for ai, bj in assignment:
            if ai < len_a and bj < len_b:
                total_cost = cost_matrix[ai][bj]
                if total_cost <= self.number_pair_threshold:
                    pairs.append(
                        self._build_pair(
                            nums_a[ai],
                            nums_b[bj],
                            total_cost,
                            penalty_matrix[ai][bj],
                            gap_matrix[ai][bj],
                        )
                    )
                    matched_a.add(ai)
                    matched_b.add(bj)

        for idx in range(len_a):
            if idx not in matched_a:
                pairs.append(self._single_side_pair(nums_a[idx], None))
        for idx in range(len_b):
            if idx not in matched_b:
                pairs.append(self._single_side_pair(None, nums_b[idx]))
        return pairs

    def _numeric_pair_cost(
        self, left: Dict[str, Any], right: Dict[str, Any]
    ) -> Tuple[float, float, float]:
        val_left = float(left["value"])
        val_right = float(right["value"])
        denom = max(abs(val_left), abs(val_right), 1.0)
        gap = abs(val_left - val_right) / denom
        penalty = 0.0
        currency_left = left.get("currency")
        currency_right = right.get("currency")
        if currency_left and currency_right and currency_left != currency_right:
            penalty += 0.6
        elif (currency_left and not currency_right) or (currency_right and not currency_left):
            penalty += 0.3
        if bool(left.get("is_percent")) != bool(right.get("is_percent")):
            penalty += 0.4
        scale_left = left.get("scale")
        scale_right = right.get("scale")
        if scale_left and scale_right and scale_left != scale_right:
            penalty += 0.2
        return gap + penalty, penalty, gap

    def _build_pair(
        self,
        left: Dict[str, Any],
        right: Dict[str, Any],
        total_cost: float,
        penalty: float,
        gap: float,
    ) -> Dict[str, Any]:
        old = float(left["value"])
        new = float(right["value"])
        delta = new - old
        relative = (delta / old) if abs(old) > 1e-12 else None
        return {
            "old": old,
            "new": new,
            "delta": delta,
            "relative": relative,
            "old_raw": left.get("raw"),
            "new_raw": right.get("raw"),
            "unit": left.get("unit") or right.get("unit"),
            "currency": left.get("currency") or right.get("currency"),
            "is_percent": left.get("is_percent") or right.get("is_percent"),
            "scale": left.get("scale") or right.get("scale"),
            "pair_cost": total_cost,
            "pair_penalty": penalty,
            "pair_gap": gap,
            "pair_status": "matched",
        }

    def _single_side_pair(
        self,
        left: Optional[Dict[str, Any]],
        right: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        unit = (left or {}).get("unit") or (right or {}).get("unit")
        currency = (left or {}).get("currency") or (right or {}).get("currency")
        scale = (left or {}).get("scale") or (right or {}).get("scale")
        is_percent = (left or {}).get("is_percent") or (right or {}).get("is_percent")
        status = "unmatched_a" if right is None else "unmatched_b"
        return {
            "old": float(left["value"]) if left else None,
            "new": float(right["value"]) if right else None,
            "delta": None,
            "relative": None,
            "old_raw": (left or {}).get("raw"),
            "new_raw": (right or {}).get("raw"),
            "unit": unit,
            "currency": currency,
            "is_percent": is_percent,
            "scale": scale,
            "pair_cost": None,
            "pair_penalty": None,
            "pair_gap": None,
            "pair_status": status,
        }

    @staticmethod
    def _hungarian(cost_matrix: List[List[float]]) -> List[Tuple[int, int]]:
        if not cost_matrix:
            return []
        n = len(cost_matrix)
        m = len(cost_matrix[0]) if cost_matrix else 0
        if n != m:
            raise ValueError("Hungarian solver expects a square cost matrix")
        size = n
        u = [0.0] * (size + 1)
        v = [0.0] * (size + 1)
        p = [0] * (size + 1)
        way = [0] * (size + 1)
        for i in range(1, size + 1):
            p[0] = i
            minv = [float("inf")] * (size + 1)
            used = [False] * (size + 1)
            j0 = 0
            while True:
                used[j0] = True
                i0 = p[j0]
                delta = float("inf")
                j1 = 0
                for j in range(1, size + 1):
                    if used[j]:
                        continue
                    cur = cost_matrix[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
                for j in range(size + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta
                j0 = j1
                if p[j0] == 0:
                    break
            while True:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
                if j0 == 0:
                    break
        assignment: List[Tuple[int, int]] = []
        for j in range(1, size + 1):
            if p[j]:
                assignment.append((p[j] - 1, j - 1))
        return assignment

    @staticmethod
    def _normalize_numeric_token(raw: str) -> Optional[Dict[str, Any]]:
        if not raw:
            return None
        text = raw.strip()
        if not text:
            return None
        text = text.replace("（", "(").replace("）", ")")
        sign = 1.0
        if text.startswith("(") and text.endswith(")"):
            sign *= -1.0
            text = text[1:-1].strip()
        if text.startswith("+"):
            text = text[1:].strip()
        elif text.startswith("-"):
            sign *= -1.0
            text = text[1:].strip()

        text, prefix_currency = SimpleTextDiffer._strip_currency_prefix(text)
        currency = prefix_currency
        text, suffix_currency = SimpleTextDiffer._strip_currency_suffix(text)
        if suffix_currency and not currency:
            currency = suffix_currency
        text, is_percent = SimpleTextDiffer._strip_percent_suffix(text)
        unit = "percent" if is_percent else None
        text, scale_label, scale = SimpleTextDiffer._strip_scale_suffix(text)

        normalized = text.replace(",", "").replace(" ", "")
        if not normalized:
            return None
        try:
            value = float(normalized)
        except ValueError:
            return None
        value *= sign * scale
        if currency and not unit:
            unit = currency
        return {
            "raw": raw.strip(),
            "value": value,
            "currency": currency,
            "unit": unit,
            "is_percent": bool(is_percent),
            "scale": scale_label,
        }

    @staticmethod
    def _strip_currency_prefix(text: str) -> Tuple[str, Optional[str]]:
        if not text:
            return "", None
        stripped = text
        detected = None
        while True:
            stripped = stripped.lstrip()
            if stripped and stripped[0] in _CURRENCY_SYMBOLS:
                detected = detected or _CURRENCY_SYMBOLS[stripped[0]]
                stripped = stripped[1:]
                continue
            matched = False
            upper = stripped.upper()
            for prefix in sorted(_CURRENCY_WORD_PREFIXES.keys(), key=len, reverse=True):
                if upper.startswith(prefix):
                    next_idx = len(prefix)
                    if len(stripped) == next_idx or not stripped[next_idx].isalpha():
                        detected = detected or _CURRENCY_WORD_PREFIXES[prefix]
                        stripped = stripped[next_idx:]
                        matched = True
                        break
            if not matched:
                break
        return stripped.strip(), detected

    @staticmethod
    def _strip_currency_suffix(text: str) -> Tuple[str, Optional[str]]:
        stripped = text.rstrip()
        lowered = stripped.lower()
        for suffix in sorted(_CURRENCY_SUFFIXES.keys(), key=len, reverse=True):
            key = suffix.lower()
            if lowered.endswith(key):
                new_text = stripped[: -len(suffix)].rstrip()
                return new_text, _CURRENCY_SUFFIXES[suffix]
        return stripped, None

    @staticmethod
    def _strip_percent_suffix(text: str) -> Tuple[str, bool]:
        stripped = text.rstrip()
        for suffix in ("%", "％"):
            if stripped.endswith(suffix):
                return stripped[: -len(suffix)].rstrip(), True
        return stripped, False

    @staticmethod
    def _strip_scale_suffix(text: str) -> Tuple[str, Optional[str], float]:
        stripped = text.rstrip()
        lower = stripped.lower()
        for suffix, factor, casefold in _SCALE_SUFFIXES:
            key = suffix.lower() if casefold else suffix
            target = lower if casefold else stripped
            if target.endswith(key):
                new_text = stripped[: -len(key)].rstrip()
                return new_text, suffix, factor
        return stripped, None, 1.0

    @staticmethod
    def _read_lines(path: Path) -> List[str]:
        return Path(path).read_text(encoding="utf-8").splitlines()

    def _context_radius_for_metadata(self) -> int:
        return self.context_lines if self.context_lines > 0 else 2

    def _gather_context(
        self,
        lines: Sequence[str],
        line_no: Optional[int],
        radius: int,
    ) -> List[Dict[str, Any]]:
        if line_no is None or not lines or radius <= 0:
            return []
        start = max(1, line_no - radius)
        end = min(len(lines), line_no + radius)
        context: List[Dict[str, Any]] = []
        for idx in range(start, end + 1):
            if idx == line_no:
                continue
            context.append({"line": idx, "text": lines[idx - 1]})
        return context

    def _context_preview(self, ctx: Optional[Sequence[Dict[str, Any]]]) -> Optional[str]:
        if not ctx:
            return None
        parts = []
        for item in ctx:
            line_no = item.get("line")
            text = _clip(item.get("text", ""))
            if text:
                prefix = f"L{line_no}: " if line_no is not None else ""
                parts.append(f"{prefix}{text}")
        return "\n".join(parts) if parts else None
