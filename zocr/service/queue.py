from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RedisQueueSettings:
    url: str
    stream: str
    group: str


def redis_settings_from_env() -> RedisQueueSettings:
    return RedisQueueSettings(
        url=(os.environ.get("ZOCR_API_REDIS_URL") or "redis://redis:6379/0").strip(),
        stream=(os.environ.get("ZOCR_API_REDIS_STREAM") or "zocr_jobs").strip(),
        group=(os.environ.get("ZOCR_API_REDIS_GROUP") or "zocr_workers").strip(),
    )


def build_queue_backend() -> str:
    backend = (os.environ.get("ZOCR_API_QUEUE_BACKEND") or "inline").strip().lower()
    if backend in {"inline", "inproc", "local"}:
        return "inline"
    return backend


class RedisStreamQueue:
    def __init__(self, *, url: str, stream: str):
        try:
            import redis  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Redis queue requires `redis`. Install with `pip install -e '.[api_redis]'`."
            ) from exc
        self.stream = stream
        self._client = redis.Redis.from_url(url)
        self._response_error = getattr(redis, "ResponseError", Exception)

    def ensure_consumer_group(self, group: str) -> None:
        try:
            self._client.xgroup_create(self.stream, group, id="0-0", mkstream=True)
        except Exception as exc:
            if isinstance(exc, self._response_error) and "BUSYGROUP" in str(exc):
                return
            raise

    def enqueue(self, job_id: str, tenant_id: str) -> str:
        message_id = self._client.xadd(self.stream, {"job_id": job_id, "tenant_id": tenant_id})
        if isinstance(message_id, bytes):
            return message_id.decode("utf-8", "replace")
        return str(message_id)

    def read(
        self,
        *,
        group: str,
        consumer: str,
        start_id: str = ">",
        count: int = 1,
        block_ms: int = 5000,
    ) -> list[tuple[str, Any]]:
        response = self._client.xreadgroup(
            group,
            consumer,
            {self.stream: start_id},
            count=max(1, int(count)),
            block=max(0, int(block_ms)),
        )
        items: list[tuple[str, Any]] = []
        for _stream_name, messages in response or []:
            for message_id, payload in messages:
                if isinstance(message_id, bytes):
                    mid = message_id.decode("utf-8", "replace")
                else:
                    mid = str(message_id)
                items.append((mid, payload))
        return items

    def ack(self, *, group: str, message_id: str) -> int:
        return int(self._client.xack(self.stream, group, message_id))

    def claim_stale(
        self,
        *,
        group: str,
        consumer: str,
        min_idle_ms: int,
        start_id: str = "0-0",
        count: int = 10,
    ) -> tuple[str, list[tuple[str, Any]]]:
        response = self._client.xautoclaim(
            name=self.stream,
            groupname=group,
            consumername=consumer,
            min_idle_time=max(1, int(min_idle_ms)),
            start_id=start_id,
            count=max(1, int(count)),
        )
        next_start = "0-0"
        items: list[tuple[str, Any]] = []
        if not isinstance(response, (list, tuple)) or not response:
            return next_start, items
        raw_next = response[0]
        if isinstance(raw_next, bytes):
            next_start = raw_next.decode("utf-8", "replace")
        elif raw_next is not None:
            next_start = str(raw_next)
        messages = response[1] if len(response) > 1 else []
        for message_id, payload in messages or []:
            if isinstance(message_id, bytes):
                mid = message_id.decode("utf-8", "replace")
            else:
                mid = str(message_id)
            items.append((mid, payload))
        return next_start, items


def build_redis_queue() -> RedisStreamQueue:
    settings = redis_settings_from_env()
    return RedisStreamQueue(url=settings.url, stream=settings.stream)
