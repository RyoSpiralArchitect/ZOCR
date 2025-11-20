"""CLI entrypoint for ``python -m zocr.core``."""

from __future__ import annotations

from .zocr_core import main


def cli() -> None:
    """Invoke the legacy ``zocr_core`` CLI."""

    main()


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess test
    cli()
