# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024 ZOCR contributors

"""Core package shim keeping ``__init__`` minimal."""

from . import _surface as _surface

__all__ = _surface.__all__


def __getattr__(name: str):
    return getattr(_surface, name)


def __dir__() -> list[str]:
    return _surface.__dir__()
