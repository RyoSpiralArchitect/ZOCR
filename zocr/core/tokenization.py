# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024 ZOCR contributors

"""Tokenizer helpers for Japanese + Latin text."""
from __future__ import annotations

import re
from typing import List

__all__ = ["tokenize_jp"]

_token_re = re.compile(r"[A-Za-z]+|\d+(?:,\d{3})*(?:\.\d+)?")


def tokenize_jp(text: str) -> List[str]:
    text = text or ""
    toks = _token_re.findall(text)
    jp = "".join(ch for ch in text if ord(ch) > 127 and not ch.isspace())
    toks.extend(jp[i : i + 2] for i in range(max(0, len(jp) - 1)))
    return [tok.lower() for tok in toks if tok]
