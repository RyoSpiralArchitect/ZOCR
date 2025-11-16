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
        }

    def _collect_numeric_changes(
        self,
        opcodes,
        lines_a: Sequence[str],
        lines_b: Sequence[str],
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for tag, i1, i2, j1, j2 in opcodes:
            if tag not in {"replace"}:
                continue
            slice_a = list(enumerate(lines_a[i1:i2], start=i1 + 1))
            slice_b = list(enumerate(lines_b[j1:j2], start=j1 + 1))
            for (line_no_a, text_a), (line_no_b, text_b) in zip(slice_a, slice_b):
                nums_a = self._extract_numbers(text_a)
                nums_b = self._extract_numbers(text_b)
                if not nums_a and not nums_b:
                    continue
                pairs = self._pair_numbers(nums_a, nums_b)
                if not pairs and (not nums_a or not nums_b):
                    pairs = [
                        {
                            "old": nums_a[0]["value"] if nums_a else None,
                            "new": nums_b[0]["value"] if nums_b else None,
                            "delta": None,
                            "relative": None,
                            "old_raw": nums_a[0]["raw"] if nums_a else None,
                            "new_raw": nums_b[0]["raw"] if nums_b else None,
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
