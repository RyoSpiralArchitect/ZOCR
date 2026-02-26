from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser("zocr-api")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", default="info")
    env_workers = os.environ.get("ZOCR_API_WORKERS")
    try:
        default_workers = int(env_workers) if env_workers else 1
    except ValueError:
        default_workers = 1
    parser.add_argument("--workers", type=int, default=default_workers)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    args.workers = max(1, int(args.workers))
    if args.reload and args.workers > 1:
        raise SystemExit("--reload cannot be used with --workers > 1")

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
        workers=args.workers,
        factory=True,
    )
