# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024 ZOCR contributors

# -*- coding: utf-8 -*-
"""Allow `python -m zocr.diff` to invoke the CLI."""
from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    main()
