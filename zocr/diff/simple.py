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
from typing import Any, Dict, List, Sequence


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


_NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


class SimpleTextDiffer:
    """Compare two plain-text documents with git-like output."""

    def __init__(self, context_lines: int = 3) -> None:
        self.context_lines = max(0, int(context_lines))

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
                "trace_a": f"{label_a}#L{line_a}" if line_a is not None else None,
                "trace_b": f"{label_b}#L{line_b}" if line_b is not None else None,
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
                events.append(event)
        return events

    def _collect_numeric_changes(
        self,
        opcodes,
        lines_a: Sequence[str],
        lines_b: Sequence[str],
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == "equal":
                continue
            slice_a = list(enumerate(lines_a[i1:i2], start=i1 + 1))
            slice_b = list(enumerate(lines_b[j1:j2], start=j1 + 1))
            max_len = max(len(slice_a), len(slice_b))
            if max_len == 0:
                continue
            for idx in range(max_len):
                if idx < len(slice_a):
                    line_no_a, text_a = slice_a[idx]
                else:
                    line_no_a, text_a = None, ""
                if idx < len(slice_b):
                    line_no_b, text_b = slice_b[idx]
                else:
                    line_no_b, text_b = None, ""

                nums_a = self._extract_numbers(text_a)
                nums_b = self._extract_numbers(text_b)
                if not nums_a and not nums_b:
                    continue
                pairs = self._pair_numbers(nums_a, nums_b)
                if not pairs and (not nums_a or not nums_b):
                    token_a = nums_a[0] if nums_a else {"value": None, "raw": None}
                    token_b = nums_b[0] if nums_b else {"value": None, "raw": None}
                    pairs = [
                        {
                            "old": token_a["value"],
                            "new": token_b["value"],
                            "delta": None,
                            "relative": None,
                            "old_raw": token_a["raw"],
                            "new_raw": token_b["raw"],
                        }
                    ]
                results.append(
                    {
                        "line_a": line_no_a,
                        "line_b": line_no_b,
                        "text_a": text_a,
                        "text_b": text_b,
                        "numbers_a": [token["raw"] for token in nums_a],
                        "numbers_b": [token["raw"] for token in nums_b],
                        "pairs": pairs,
                    }
                )
        return results

    @staticmethod
    def _extract_numbers(text: str) -> List[Dict[str, Any]]:
        tokens: List[Dict[str, Any]] = []
        for match in _NUM_RE.finditer(text or ""):
            raw = match.group(0)
            try:
                normalized = raw.replace(",", "")
                value = float(normalized)
            except Exception:
                continue
            tokens.append({"raw": raw, "value": value})
        return tokens

    @staticmethod
    def _pair_numbers(
        nums_a: Sequence[Dict[str, Any]],
        nums_b: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        pairs: List[Dict[str, Any]] = []
        length = min(len(nums_a), len(nums_b))
        for idx in range(length):
            old = float(nums_a[idx]["value"])
            new = float(nums_b[idx]["value"])
            delta = new - old
            relative = (delta / old) if abs(old) > 1e-12 else None
            pairs.append(
                {
                    "old": old,
                    "new": new,
                    "delta": delta,
                    "relative": relative,
                    "old_raw": nums_a[idx]["raw"],
                    "new_raw": nums_b[idx]["raw"],
                }
            )
        return pairs

    @staticmethod
    def _read_lines(path: Path) -> List[str]:
        return Path(path).read_text(encoding="utf-8").splitlines()
