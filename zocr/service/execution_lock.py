from __future__ import annotations

import hashlib
import os
import uuid
from contextlib import contextmanager
from typing import Any, Iterator


@contextmanager
def execution_lock(store: Any, repository: Any, job_id: str, tenant_id: str) -> Iterator[bool]:
    key = hashlib.sha256(f"{tenant_id}:{job_id}".encode()).digest()
    if repository.backend == "postgres":
        with repository._connect() as connection:
            with connection.cursor() as cursor:
                lock_id = int.from_bytes(key[:8], "big", signed=True)
                cursor.execute("SELECT pg_try_advisory_lock(%s)", (lock_id,))
                acquired = bool(cursor.fetchone()[0])
                try:
                    yield acquired
                finally:
                    if acquired:
                        cursor.execute("SELECT pg_advisory_unlock(%s)", (lock_id,))
        return
    locks_dir = store.storage_dir / "execution-locks"
    locks_dir.mkdir(parents=True, exist_ok=True)
    with (locks_dir / f"{key.hex()}.lock").open("a+b") as stream:
        if os.name == "nt":
            import msvcrt

            stream.seek(0)
            if not stream.read(1):
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                yield False
                return
            unlock = lambda: (stream.seek(0), msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1))
        else:
            import fcntl

            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield False
                return
            unlock = lambda: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        try:
            yield True
        finally:
            unlock()


class ExecutionLost(RuntimeError):
    pass


class ExecutionGuard:
    def __init__(self, repository, job_id, tenant_id, connection=None, lock_id=None):
        self.repository = repository
        self.job_id = job_id
        self.tenant_id = tenant_id
        self.connection = connection
        self.lock_id = lock_id
        self.token = uuid.uuid4().hex
        self.started = False

    def check(self):
        if self.connection is not None:
            try:
                with self.connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT EXISTS (SELECT 1 FROM zocr_jobs WHERE tenant_id=%s AND job_id=%s "
                        "AND payload->>'execution_token'=%s) AND EXISTS "
                        "(SELECT 1 FROM pg_locks WHERE pid=pg_backend_pid() AND locktype='advisory' "
                        "AND classid=%s AND objid=%s AND objsubid=1)",
                        (self.tenant_id, self.job_id, self.token,
                         (self.lock_id >> 32) & 0xffffffff, self.lock_id & 0xffffffff),
                    )
                    valid = bool(cursor.fetchone()[0])
            except self.repository._psycopg.Error as exc:
                raise ExecutionLost("Execution database session was lost") from exc
        else:
            current = self.repository.read_job(self.job_id, self.tenant_id)
            valid = isinstance(current, dict) and current.get("execution_token") == self.token
        if not valid:
            raise ExecutionLost("Execution ownership was superseded or released")

    def write_job(self, payload):
        expected = self.token if self.started else payload.get("execution_token")
        updated = dict(payload, execution_token=self.token, tenant_id=self.tenant_id)
        if self.connection is not None:
            try:
                with self.connection.cursor() as cursor:
                    cursor.execute(
                        "UPDATE zocr_jobs SET status=%s, updated_at=NOW(), payload=%s "
                        "WHERE tenant_id=%s AND job_id=%s "
                        "AND payload->>'execution_token' IS NOT DISTINCT FROM %s "
                        "AND EXISTS (SELECT 1 FROM pg_locks WHERE pid=pg_backend_pid() "
                        "AND locktype='advisory' AND classid=%s AND objid=%s AND objsubid=1)",
                        (str(updated.get("status") or "running"), self.repository._jsonb(updated),
                         self.tenant_id, self.job_id, expected,
                         (self.lock_id >> 32) & 0xffffffff, self.lock_id & 0xffffffff),
                    )
                    if cursor.rowcount != 1:
                        raise ExecutionLost("Execution ownership was superseded or released")
            except self.repository._psycopg.Error as exc:
                raise ExecutionLost("Execution database session was lost") from exc
        else:
            current = self.repository.read_job(self.job_id, self.tenant_id)
            if not isinstance(current, dict) or current.get("execution_token") != expected:
                raise ExecutionLost("Execution ownership was superseded or deleted")
            self.repository.write_job(self.job_id, self.tenant_id, updated)
        payload.update(updated)
        self.started = True


@contextmanager
def execution_session(store, repository, job_id, tenant_id):
    if repository.backend != "postgres":
        with execution_lock(store, repository, job_id, tenant_id) as acquired:
            yield ExecutionGuard(repository, job_id, tenant_id) if acquired else None
        return
    repository._maybe_init_schema()
    lock_id = int.from_bytes(hashlib.sha256(f"{tenant_id}:{job_id}".encode()).digest()[:8], "big", signed=True)
    with repository._connect(connect_timeout=5, keepalives_idle=2, keepalives_interval=1,
                             keepalives_count=2, tcp_user_timeout=5000,
                             target_session_attrs="read-write") as connection:
        connection.autocommit = True
        connection.execute("SET statement_timeout = 2000")
        with connection.cursor() as cursor:
            cursor.execute("SELECT pg_try_advisory_lock(%s)", (lock_id,))
            acquired = bool(cursor.fetchone()[0])
        try:
            yield ExecutionGuard(repository, job_id, tenant_id, connection, lock_id) if acquired else None
        finally:
            if acquired and not connection.closed:
                try:
                    connection.execute("SELECT pg_advisory_unlock(%s)", (lock_id,))
                except repository._psycopg.Error:
                    pass
