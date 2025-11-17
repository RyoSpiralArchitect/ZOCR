"""Compatibility shim exposing the consensus runtime, CLI, and helpers."""
from __future__ import annotations

from . import cli as _cli
from . import local_search as _local_search
from . import runtime as _runtime


def _export_from(module, names=None):
    exported = []
    if names is None:
        names = getattr(module, "__all__", None)
    if names is None:
        names = [n for n in dir(module) if not (n.startswith("__") and n.endswith("__"))]
    for name in names:
        globals()[name] = getattr(module, name)
        exported.append(name)
    return exported


__all__ = []

for _mod in (_runtime, _local_search, _cli):
    __all__.extend(_export_from(_mod))


if __name__ == "__main__":  # pragma: no cover
    _cli.main()
