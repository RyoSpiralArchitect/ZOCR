"""Z-OCR package shim keeping ``__init__`` minimal."""

from . import _surface as _surface
from ._version import __version__

__all__ = [*_surface.__all__, "__version__"]


def __getattr__(name: str):
    return getattr(_surface, name)


def __dir__() -> list[str]:
    return sorted(set(_surface.__dir__() + ["__version__"]))
