# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024 ZOCR contributors

"""Pytest configuration shared across the suite."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
