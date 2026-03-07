from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Optional, Protocol
from urllib.request import Request as UrlRequest
from urllib.request import urlopen


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    limit: int
    remaining: int
    retry_after_sec: int


@dataclass(frozen=True)
class RedisRateLimitSettings:
    url: str
    prefix: str
    window_sec: int


@dataclass(frozen=True)
class AuditSettings:
    backends: tuple[str, ...]
    read_backend: Optional[str]
    file_path: Optional[str]
    http_url: Optional[str]
    http_timeout_sec: float


@dataclass(frozen=True)
class TenantApprovalNotificationSettings:
    enabled: bool
    url: Optional[str]
    timeout_sec: float


class RateLimiter(Protocol):
    limit_per_min: int
    window_sec: int

    @property
    def enabled(self) -> bool: ...

    def build_key(
        self,
        *,
        tenant_id: Optional[str],
        subject: Optional[str],
        client: Optional[str],
    ) -> str: ...

    def check(
        self,
        key: str,
        *,
        limit_per_min: Optional[int] = None,
        now: Optional[float] = None,
    ) -> RateLimitResult: ...


class AuditSink(Protocol):
    backend: str

    def write(self, event: str, payload: dict[str, Any]) -> None: ...


class AuditSearchableSink(Protocol):
    backend: str

    def search(
        self,
        *,
        limit: int = 100,
        tenant_id: Optional[str] = None,
        event: Optional[str] = None,
        subject: Optional[str] = None,
        request_id: Optional[str] = None,
        contains: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> list[dict[str, Any]]: ...


class TenantApprovalNotifier(Protocol):
    enabled: bool

    def notify(self, event: str, payload: dict[str, Any]) -> None: ...


class _BaseRateLimiter:
    def __init__(self, *, limit_per_min: int = 0, window_sec: int = 60):
        self.limit_per_min = max(0, int(limit_per_min))
        self.window_sec = max(1, int(window_sec))

    @property
    def enabled(self) -> bool:
        return self.limit_per_min > 0

    def build_key(
        self,
        *,
        tenant_id: Optional[str],
        subject: Optional[str],
        client: Optional[str],
    ) -> str:
        tenant_part = (tenant_id or "_").strip() or "_"
        subject_part = (subject or "_").strip() or "_"
        client_part = (client or "_").strip() or "_"
        return f"{tenant_part}:{subject_part}:{client_part}"

    def _effective_limit(self, value: Optional[int]) -> int:
        if value is None:
            return self.limit_per_min
        return max(0, int(value))


class InMemoryRateLimiter(_BaseRateLimiter):
    def __init__(self, *, limit_per_min: int = 0, window_sec: int = 60):
        super().__init__(limit_per_min=limit_per_min, window_sec=window_sec)
        self._lock = Lock()
        self._buckets: dict[str, deque[float]] = {}

    def check(
        self,
        key: str,
        *,
        limit_per_min: Optional[int] = None,
        now: Optional[float] = None,
    ) -> RateLimitResult:
        effective_limit = self._effective_limit(limit_per_min)
        if effective_limit <= 0:
            return RateLimitResult(allowed=True, limit=0, remaining=0, retry_after_sec=0)

        ts = time.time() if now is None else float(now)
        window_floor = ts - float(self.window_sec)
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = deque()
                self._buckets[key] = bucket
            while bucket and bucket[0] <= window_floor:
                bucket.popleft()
            if len(bucket) >= effective_limit:
                retry_after = max(1, int(bucket[0] + float(self.window_sec) - ts))
                return RateLimitResult(
                    allowed=False,
                    limit=effective_limit,
                    remaining=0,
                    retry_after_sec=retry_after,
                )
            bucket.append(ts)
            remaining = max(0, effective_limit - len(bucket))
            return RateLimitResult(
                allowed=True,
                limit=effective_limit,
                remaining=remaining,
                retry_after_sec=0,
            )


class RedisRateLimiter(_BaseRateLimiter):
    def __init__(
        self,
        *,
        url: str,
        prefix: str,
        limit_per_min: int = 0,
        window_sec: int = 60,
    ):
        super().__init__(limit_per_min=limit_per_min, window_sec=window_sec)
        try:
            import redis  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Redis rate limiter requires `redis`. Install with `pip install -e '.[api_redis]'`."
            ) from exc
        redis_url = str(url or "").strip()
        if not redis_url:
            raise RuntimeError(
                "ZOCR_API_RATE_LIMIT_REDIS_URL or ZOCR_API_REDIS_URL is required when "
                "ZOCR_API_RATE_LIMIT_BACKEND=redis."
            )
        self.prefix = str(prefix or "zocr_rate_limit").strip() or "zocr_rate_limit"
        self._client = redis.Redis.from_url(redis_url)

    def check(
        self,
        key: str,
        *,
        limit_per_min: Optional[int] = None,
        now: Optional[float] = None,
    ) -> RateLimitResult:
        effective_limit = self._effective_limit(limit_per_min)
        if effective_limit <= 0:
            return RateLimitResult(allowed=True, limit=0, remaining=0, retry_after_sec=0)

        ts = time.time() if now is None else float(now)
        slot = int(ts // float(self.window_sec))
        window_key = f"{self.prefix}:{slot}:{key}"
        count = int(self._client.incr(window_key))
        if count == 1:
            self._client.expire(window_key, self.window_sec + 1)
        if count > effective_limit:
            retry_after = max(1, int(((slot + 1) * self.window_sec) - ts))
            return RateLimitResult(
                allowed=False,
                limit=effective_limit,
                remaining=0,
                retry_after_sec=retry_after,
            )
        remaining = max(0, effective_limit - count)
        return RateLimitResult(
            allowed=True,
            limit=effective_limit,
            remaining=remaining,
            retry_after_sec=0,
        )


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _normalized_rate_limit_backend(value: Optional[str]) -> str:
    backend = str(value or "local").strip().lower()
    if backend in {"local", "memory", "mem", "in_memory", "in-memory"}:
        return "local"
    if backend in {"redis", "shared"}:
        return "redis"
    raise RuntimeError(
        "Unsupported ZOCR_API_RATE_LIMIT_BACKEND=%r. Supported backends: local, redis."
        % (value,)
    )


def redis_rate_limit_settings_from_env(*, default_url: Optional[str] = None) -> RedisRateLimitSettings:
    redis_url = (
        os.environ.get("ZOCR_API_RATE_LIMIT_REDIS_URL")
        or default_url
        or os.environ.get("ZOCR_API_REDIS_URL")
        or "redis://redis:6379/0"
    )
    return RedisRateLimitSettings(
        url=str(redis_url).strip(),
        prefix=(os.environ.get("ZOCR_API_RATE_LIMIT_REDIS_PREFIX") or "zocr_rate_limit").strip()
        or "zocr_rate_limit",
        window_sec=max(1, _env_int("ZOCR_API_RATE_LIMIT_WINDOW_SEC", 60)),
    )


def build_rate_limiter(
    *,
    limit_per_min: int,
    default_redis_url: Optional[str] = None,
) -> tuple[str, RateLimiter]:
    backend = _normalized_rate_limit_backend(os.environ.get("ZOCR_API_RATE_LIMIT_BACKEND") or "local")
    settings = redis_rate_limit_settings_from_env(default_url=default_redis_url)
    if backend == "redis":
        return (
            backend,
            RedisRateLimiter(
                url=settings.url,
                prefix=settings.prefix,
                limit_per_min=limit_per_min,
                window_sec=settings.window_sec,
            ),
        )
    return (
        backend,
        InMemoryRateLimiter(limit_per_min=limit_per_min, window_sec=settings.window_sec),
    )


def _audit_record(event: str, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    record = {"ts": ts, "event": str(event), **payload}
    return ts, record


def _parse_audit_timestamp(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        raw = value.strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(raw)
        except Exception:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _matches_audit_filters(
    record: dict[str, Any],
    *,
    tenant_id: Optional[str],
    event: Optional[str],
    subject: Optional[str],
    request_id: Optional[str],
    contains: Optional[str],
    since: Optional[datetime],
    until: Optional[datetime],
) -> bool:
    if tenant_id and str(record.get("tenant_id") or "") != str(tenant_id):
        return False
    if event and str(record.get("event") or "") != str(event):
        return False
    if subject and str(record.get("subject") or "") != str(subject):
        return False
    if request_id and str(record.get("request_id") or "") != str(request_id):
        return False
    occurred_at = _parse_audit_timestamp(record.get("ts"))
    if since is not None and (occurred_at is None or occurred_at < since):
        return False
    if until is not None and (occurred_at is None or occurred_at > until):
        return False
    if contains:
        haystack = json.dumps(record, ensure_ascii=False).lower()
        if contains.lower() not in haystack:
            return False
    return True


class FileAuditSink:
    backend = "file"

    def __init__(self, path: str):
        raw_path = str(path or "").strip()
        if not raw_path:
            raise RuntimeError("ZOCR_API_AUDIT_LOG_PATH is required when audit sink `file` is enabled.")
        self.path = Path(raw_path).expanduser()
        self._lock = Lock()

    def write(self, event: str, payload: dict[str, Any]) -> None:
        _ts, record = _audit_record(event, payload)
        line = json.dumps(record, ensure_ascii=False)
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
                    handle.write("\n")
        except Exception:
            return

    def search(
        self,
        *,
        limit: int = 100,
        tenant_id: Optional[str] = None,
        event: Optional[str] = None,
        subject: Optional[str] = None,
        request_id: Optional[str] = None,
        contains: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return []
        items: list[dict[str, Any]] = []
        max_items = max(1, int(limit))
        contains_filter = contains.lower() if contains else None
        for line in reversed(lines):
            if not line.strip():
                continue
            if contains_filter and contains_filter not in line.lower():
                continue
            try:
                parsed = json.loads(line)
            except Exception:
                continue
            if not isinstance(parsed, dict):
                continue
            if not _matches_audit_filters(
                parsed,
                tenant_id=tenant_id,
                event=event,
                subject=subject,
                request_id=request_id,
                contains=contains,
                since=since,
                until=until,
            ):
                continue
            items.append(parsed)
            if len(items) >= max_items:
                break
        return items


class PostgresAuditSink:
    backend = "postgres"

    def __init__(self, *, dsn: str, auto_init: bool = True):
        try:
            import psycopg  # type: ignore
            from psycopg.types.json import Jsonb  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Postgres audit sink requires `psycopg`. Install with `pip install -e '.[api_db]'`."
            ) from exc
        self._psycopg = psycopg
        self._jsonb = Jsonb
        self._dsn = str(dsn or "").strip()
        self._auto_init = bool(auto_init)
        self._lock = Lock()
        self._initialized = False
        if not self._dsn:
            raise RuntimeError(
                "ZOCR_API_AUDIT_DATABASE_URL or ZOCR_API_DATABASE_URL is required when audit sink "
                "`postgres` is enabled."
            )

    def _connect(self):
        return self._psycopg.connect(self._dsn)

    def _maybe_init_schema(self) -> None:
        if not self._auto_init or self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            self._init_schema()
            self._initialized = True

    def _init_schema(self) -> None:
        statements = [
            """
            CREATE TABLE IF NOT EXISTS zocr_audit_events (
                id BIGSERIAL PRIMARY KEY,
                occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                event TEXT NOT NULL,
                tenant_id TEXT,
                subject TEXT,
                request_id TEXT,
                payload JSONB NOT NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS zocr_audit_events_occurred_idx ON zocr_audit_events (occurred_at DESC)",
            "CREATE INDEX IF NOT EXISTS zocr_audit_events_tenant_idx ON zocr_audit_events (tenant_id, occurred_at DESC)",
            "CREATE INDEX IF NOT EXISTS zocr_audit_events_request_idx ON zocr_audit_events (request_id)",
        ]
        with self._connect() as conn:
            with conn.cursor() as cur:
                for statement in statements:
                    cur.execute(statement)
            conn.commit()

    def write(self, event: str, payload: dict[str, Any]) -> None:
        try:
            self._maybe_init_schema()
            ts, record = _audit_record(event, payload)
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO zocr_audit_events (
                            occurred_at,
                            event,
                            tenant_id,
                            subject,
                            request_id,
                            payload
                        ) VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (
                            ts,
                            str(event),
                            record.get("tenant_id"),
                            record.get("subject"),
                            record.get("request_id"),
                            self._jsonb(record),
                        ),
                    )
                conn.commit()
        except Exception:
            return

    def search(
        self,
        *,
        limit: int = 100,
        tenant_id: Optional[str] = None,
        event: Optional[str] = None,
        subject: Optional[str] = None,
        request_id: Optional[str] = None,
        contains: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> list[dict[str, Any]]:
        self._maybe_init_schema()
        sql = """
            SELECT occurred_at, event, tenant_id, subject, request_id, payload
            FROM zocr_audit_events
        """
        clauses: list[str] = []
        params: list[Any] = []
        if tenant_id:
            clauses.append("tenant_id = %s")
            params.append(str(tenant_id))
        if event:
            clauses.append("event = %s")
            params.append(str(event))
        if subject:
            clauses.append("subject = %s")
            params.append(str(subject))
        if request_id:
            clauses.append("request_id = %s")
            params.append(str(request_id))
        if since is not None:
            clauses.append("occurred_at >= %s")
            params.append(since)
        if until is not None:
            clauses.append("occurred_at <= %s")
            params.append(until)
        if contains:
            clauses.append("payload::text ILIKE %s")
            params.append(f"%{contains}%")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY occurred_at DESC LIMIT %s"
        params.append(max(1, int(limit)))
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    rows = cur.fetchall()
        except Exception:
            return []
        items: list[dict[str, Any]] = []
        for occurred_at, event_name, row_tenant_id, row_subject, row_request_id, payload in rows:
            record = payload if isinstance(payload, dict) else {}
            if not isinstance(record, dict):
                record = {}
            if "ts" not in record:
                if isinstance(occurred_at, datetime):
                    record["ts"] = occurred_at.astimezone(timezone.utc).replace(microsecond=0).isoformat()
                elif occurred_at is not None:
                    record["ts"] = str(occurred_at)
            record.setdefault("event", event_name)
            if row_tenant_id is not None:
                record.setdefault("tenant_id", row_tenant_id)
            if row_subject is not None:
                record.setdefault("subject", row_subject)
            if row_request_id is not None:
                record.setdefault("request_id", row_request_id)
            items.append(record)
        return items


class HttpAuditSink:
    backend = "http"

    def __init__(self, *, url: str, headers: Optional[dict[str, str]] = None, timeout_sec: float = 2.0):
        self.url = str(url or "").strip()
        if not self.url:
            raise RuntimeError("ZOCR_API_AUDIT_HTTP_URL is required when audit sink `http` is enabled.")
        self.headers = dict(headers or {})
        self.timeout_sec = max(0.1, float(timeout_sec))

    def write(self, event: str, payload: dict[str, Any]) -> None:
        _ts, record = _audit_record(event, payload)
        body = json.dumps(record, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", **self.headers}
        request = UrlRequest(self.url, data=body, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self.timeout_sec) as response:
                response.read(1)
        except Exception:
            return


class NoopTenantApprovalNotifier:
    enabled = False

    def notify(self, event: str, payload: dict[str, Any]) -> None:
        return


class HttpTenantApprovalNotifier:
    enabled = True

    def __init__(self, *, url: str, headers: Optional[dict[str, str]] = None, timeout_sec: float = 2.0):
        self.url = str(url or "").strip()
        if not self.url:
            raise RuntimeError(
                "ZOCR_API_TENANT_APPROVAL_NOTIFY_URL is required when tenant approval notifications are enabled."
            )
        self.headers = dict(headers or {})
        self.timeout_sec = max(0.1, float(timeout_sec))

    def notify(self, event: str, payload: dict[str, Any]) -> None:
        ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        record = {"ts": ts, "event": str(event), **payload}
        body = json.dumps(record, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", **self.headers}
        request = UrlRequest(self.url, data=body, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self.timeout_sec) as response:
                response.read(1)
        except Exception:
            return


class AuditLogger:
    def __init__(
        self,
        sinks: tuple[AuditSink, ...],
        *,
        read_backend: Optional[str] = None,
        read_sink: Optional[AuditSearchableSink] = None,
    ):
        self._sinks = tuple(sinks)
        self._read_backend = str(read_backend).strip() or None if read_backend else None
        self._read_sink = read_sink
        self.path = next((sink.path for sink in self._sinks if isinstance(sink, FileAuditSink)), None)

    @property
    def enabled(self) -> bool:
        return bool(self._sinks)

    @property
    def backends(self) -> tuple[str, ...]:
        return tuple(str(sink.backend) for sink in self._sinks)

    @property
    def read_backend(self) -> Optional[str]:
        return self._read_backend

    @property
    def readable(self) -> bool:
        return self._read_sink is not None

    def write(self, event: str, payload: dict[str, Any]) -> None:
        for sink in self._sinks:
            sink.write(event, payload)

    def search(
        self,
        *,
        limit: int = 100,
        tenant_id: Optional[str] = None,
        event: Optional[str] = None,
        subject: Optional[str] = None,
        request_id: Optional[str] = None,
        contains: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> list[dict[str, Any]]:
        if self._read_sink is None:
            return []
        return self._read_sink.search(
            limit=limit,
            tenant_id=tenant_id,
            event=event,
            subject=subject,
            request_id=request_id,
            contains=contains,
            since=since,
            until=until,
        )


def _normalized_audit_backends() -> tuple[str, ...]:
    raw = (os.environ.get("ZOCR_API_AUDIT_SINKS") or "").strip()
    if not raw:
        return ("file",) if (os.environ.get("ZOCR_API_AUDIT_LOG_PATH") or "").strip() else ()
    items: list[str] = []
    seen: set[str] = set()
    for part in raw.replace(";", ",").split(","):
        item = part.strip().lower()
        if not item:
            continue
        if item in {"none", "off", "disabled"}:
            return ()
        if item in {"postgresql", "pg"}:
            item = "postgres"
        elif item in {"siem", "webhook"}:
            item = "http"
        elif item not in {"file", "postgres", "http"}:
            raise RuntimeError(
                "Unsupported ZOCR_API_AUDIT_SINKS entry=%r. Supported sinks: file, postgres, http."
                % (part,)
            )
        if item not in seen:
            seen.add(item)
            items.append(item)
    return tuple(items)


def _normalized_audit_read_backend(backends: tuple[str, ...]) -> Optional[str]:
    raw = (os.environ.get("ZOCR_API_AUDIT_READ_BACKEND") or "auto").strip().lower()
    if raw in {"", "auto"}:
        for candidate in ("postgres", "file"):
            if candidate in backends:
                return candidate
        return None
    if raw in {"none", "off", "disabled"}:
        return None
    if raw in {"postgresql", "pg"}:
        return "postgres"
    if raw in {"file"}:
        return "file"
    raise RuntimeError(
        "Unsupported ZOCR_API_AUDIT_READ_BACKEND=%r. Supported backends: auto, none, file, postgres."
        % (raw,)
    )


def _load_audit_http_headers() -> dict[str, str]:
    raw = (os.environ.get("ZOCR_API_AUDIT_HTTP_HEADERS_JSON") or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("ZOCR_API_AUDIT_HTTP_HEADERS_JSON must be valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("ZOCR_API_AUDIT_HTTP_HEADERS_JSON must be a JSON object.")
    return {str(key): str(value) for key, value in parsed.items()}


def _load_tenant_approval_notify_headers() -> dict[str, str]:
    raw = (os.environ.get("ZOCR_API_TENANT_APPROVAL_NOTIFY_HEADERS_JSON") or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("ZOCR_API_TENANT_APPROVAL_NOTIFY_HEADERS_JSON must be valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("ZOCR_API_TENANT_APPROVAL_NOTIFY_HEADERS_JSON must be a JSON object.")
    return {str(key): str(value) for key, value in parsed.items()}


def build_audit_logger(*, default_dsn: Optional[str] = None) -> tuple[AuditSettings, AuditLogger]:
    backends = _normalized_audit_backends()
    read_backend = _normalized_audit_read_backend(backends)
    file_path = str(os.environ.get("ZOCR_API_AUDIT_LOG_PATH") or "").strip() or None
    http_url = str(os.environ.get("ZOCR_API_AUDIT_HTTP_URL") or "").strip() or None
    http_timeout_sec = max(0.1, _env_float("ZOCR_API_AUDIT_HTTP_TIMEOUT_SEC", 2.0))
    sinks: list[AuditSink] = []
    searchable_sinks: dict[str, AuditSearchableSink] = {}
    for backend in backends:
        if backend == "file":
            sink = FileAuditSink(file_path or "")
            sinks.append(sink)
            searchable_sinks[backend] = sink
            continue
        if backend == "postgres":
            sink = PostgresAuditSink(
                dsn=(
                    os.environ.get("ZOCR_API_AUDIT_DATABASE_URL")
                    or default_dsn
                    or os.environ.get("ZOCR_API_DATABASE_URL")
                    or ""
                ),
                auto_init=_env_truthy("ZOCR_API_AUDIT_DB_AUTO_INIT", True),
            )
            sinks.append(sink)
            searchable_sinks[backend] = sink
            continue
        if backend == "http":
            sinks.append(
                HttpAuditSink(
                    url=http_url or "",
                    headers=_load_audit_http_headers(),
                    timeout_sec=http_timeout_sec,
                )
            )
    read_sink = searchable_sinks.get(read_backend) if read_backend else None
    if read_backend == "file" and read_sink is None and file_path:
        read_sink = FileAuditSink(file_path)
    if read_backend == "postgres" and read_sink is None:
        read_sink = PostgresAuditSink(
            dsn=(
                os.environ.get("ZOCR_API_AUDIT_DATABASE_URL")
                or default_dsn
                or os.environ.get("ZOCR_API_DATABASE_URL")
                or ""
            ),
            auto_init=_env_truthy("ZOCR_API_AUDIT_DB_AUTO_INIT", True),
        )
    settings = AuditSettings(
        backends=backends,
        read_backend=read_backend,
        file_path=file_path,
        http_url=http_url,
        http_timeout_sec=http_timeout_sec,
    )
    return settings, AuditLogger(tuple(sinks), read_backend=read_backend, read_sink=read_sink)


def build_tenant_approval_notifier() -> tuple[TenantApprovalNotificationSettings, TenantApprovalNotifier]:
    notify_url = str(os.environ.get("ZOCR_API_TENANT_APPROVAL_NOTIFY_URL") or "").strip() or None
    timeout_sec = max(0.1, _env_float("ZOCR_API_TENANT_APPROVAL_NOTIFY_TIMEOUT_SEC", 2.0))
    settings = TenantApprovalNotificationSettings(
        enabled=bool(notify_url),
        url=notify_url,
        timeout_sec=timeout_sec,
    )
    if not notify_url:
        return settings, NoopTenantApprovalNotifier()
    return (
        settings,
        HttpTenantApprovalNotifier(
            url=notify_url,
            headers=_load_tenant_approval_notify_headers(),
            timeout_sec=timeout_sec,
        ),
    )
