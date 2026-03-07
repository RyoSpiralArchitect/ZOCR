from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from .jobs import process_job
from .metadata import build_job_repository, default_tenant_from_env
from .queue import build_redis_queue, redis_settings_from_env
from .storage import build_job_store


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _decode_message(payload: Any) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(payload, dict):
        return None, None
    raw = payload.get("job_id", payload.get(b"job_id"))
    raw_tenant = payload.get("tenant_id", payload.get(b"tenant_id"))
    if raw is None:
        return None, None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    if isinstance(raw_tenant, bytes):
        raw_tenant = raw_tenant.decode("utf-8", "replace")
    job_id = str(raw).strip() or None
    tenant_id = str(raw_tenant).strip() if raw_tenant is not None else None
    return job_id, (tenant_id or None)


def _log(logger: logging.Logger, fmt: str, event: str, payload: Dict[str, Any], *, level: str = "info") -> None:
    record = {"ts": _utc_now_iso(), "event": event, **payload}
    message = json.dumps(record, ensure_ascii=False) if fmt == "json" else f"{record['ts']} {event} {payload}"
    getattr(logger, level, logger.info)(message)


def main() -> None:
    parser = argparse.ArgumentParser("zocr-worker")
    parser.add_argument("--consumer", default=os.environ.get("ZOCR_WORKER_CONSUMER") or "")
    parser.add_argument("--block-ms", type=int, default=int(os.environ.get("ZOCR_WORKER_BLOCK_MS") or "5000"))
    parser.add_argument(
        "--claim-idle-ms",
        type=int,
        default=int(os.environ.get("ZOCR_WORKER_CLAIM_IDLE_MS") or "60000"),
    )
    parser.add_argument(
        "--claim-count",
        type=int,
        default=int(os.environ.get("ZOCR_WORKER_CLAIM_COUNT") or "10"),
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--drain-pending", action="store_true")
    args = parser.parse_args()

    consumer = (args.consumer or "").strip() or f"{socket.gethostname()}-{os.getpid()}"
    log_format = (os.environ.get("ZOCR_WORKER_LOG_FORMAT") or os.environ.get("ZOCR_API_LOG_FORMAT") or "json").strip().lower()
    log_level = (os.environ.get("ZOCR_WORKER_LOG_LEVEL") or os.environ.get("ZOCR_API_LOG_LEVEL") or "INFO").strip().upper()

    logger = logging.getLogger("zocr.worker")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    logger.propagate = False

    from zocr.orchestrator.zocr_pipeline import run_full_pipeline

    storage_backend, job_store = build_job_store()
    metadata_backend, job_repository = build_job_repository(job_store)
    queue_settings = redis_settings_from_env()
    queue = build_redis_queue()
    queue.ensure_consumer_group(queue_settings.group)
    default_tenant = default_tenant_from_env()

    _log(
        logger,
        log_format,
        "worker_started",
            {
                "consumer": consumer,
            "queue_backend": "redis",
            "stream": queue_settings.stream,
            "group": queue_settings.group,
            "claim_idle_ms": max(0, int(args.claim_idle_ms)),
            "storage_backend": storage_backend,
            "metadata_backend": metadata_backend,
            "storage_dir": str(job_store.storage_dir),
        },
    )

    def handle(message_id: str, payload: Any) -> None:
        job_id, tenant_id = _decode_message(payload)
        if not job_id:
            queue.ack(group=queue_settings.group, message_id=message_id)
            return
        tenant_id = tenant_id or default_tenant or "default"
        status, error, duration = process_job(
            job_id=job_id,
            tenant_id=tenant_id,
            job_store=job_store,
            job_repository=job_repository,
            run_full_pipeline=run_full_pipeline,
        )
        queue.ack(group=queue_settings.group, message_id=message_id)
        _log(
            logger,
            log_format,
            "job_processed",
            {
                "job_id": job_id,
                "tenant_id": tenant_id,
                "status": status,
                "duration_ms": int(round(duration * 1000.0)),
                "error": error,
            },
            level="info" if status == "succeeded" else "error",
        )

    if args.drain_pending:
        while True:
            pending = queue.read(
                group=queue_settings.group,
                consumer=consumer,
                start_id="0",
                count=10,
                block_ms=0,
            )
            if not pending:
                break
            for message_id, payload in pending:
                handle(message_id, payload)
                if args.once:
                    return

    try:
        while True:
            if args.claim_idle_ms > 0:
                _next_cursor, claimed = queue.claim_stale(
                    group=queue_settings.group,
                    consumer=consumer,
                    min_idle_ms=args.claim_idle_ms,
                    start_id="0-0",
                    count=args.claim_count,
                )
                for message_id, payload in claimed:
                    handle(message_id, payload)
                    if args.once:
                        return
                if claimed:
                    continue
            messages = queue.read(
                group=queue_settings.group,
                consumer=consumer,
                start_id=">",
                count=1,
                block_ms=args.block_ms,
            )
            if not messages:
                continue
            for message_id, payload in messages:
                handle(message_id, payload)
                if args.once:
                    return
    except KeyboardInterrupt:
        _log(logger, log_format, "worker_stopped", {"consumer": consumer})
