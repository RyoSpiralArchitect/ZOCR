from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser("zocr-api")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    try:
        import uvicorn  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "uvicorn is not installed. Install with `pip install -e '.[api]'` "
            "(or `pip install 'zocr-suite[api]'`)."
        ) from exc

    uvicorn.run(
        "zocr.service.app:create_app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=args.reload,
        factory=True,
    )

