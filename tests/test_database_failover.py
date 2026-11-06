import json
import os
import subprocess
import time
import uuid

import pytest

from zocr.service.execution_lock import ExecutionLost, execution_session
from zocr.service.jobs import process_job
from zocr.service.metadata import PostgresJobRepository
from zocr.service.storage import LocalJobStore, result_paths


def docker(*args):
    return subprocess.run(["docker", *args], check=True, capture_output=True, text=True, timeout=90).stdout.strip()


def wait_for(predicate):
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        try:
            value = predicate()
            if value:
                return value
        except Exception:
            pass
        time.sleep(.2)
    pytest.fail("Dedicated failover service did not become ready")


def test_streaming_replica_promotion_fences_old_publisher(tmp_path):
    if os.environ.get("ZOCR_TEST_DOCKER_FAILOVER") != "1":
        pytest.skip("Set ZOCR_TEST_DOCKER_FAILOVER=1 for isolated Docker replication/promotion")
    prefix = "zocr-failover-" + uuid.uuid4().hex[:12]
    network, primary, replica = prefix + "-net", prefix + "-primary", prefix + "-replica"
    hba = tmp_path / "pg_hba.conf"
    hba.write_text("local all all trust\nhost all all 0.0.0.0/0 scram-sha-256\n"
                   "host replication all 0.0.0.0/0 scram-sha-256\n", encoding="utf-8")
    created = []
    docker("network", "create", network)
    try:
        common = ["--network", network, "-p", "127.0.0.1::5432", "-v", f"{hba}:/etc/zocr-pg_hba.conf:ro"]
        docker("run", "-d", "--name", primary, "--network-alias", "primary", *common,
               "-e", "POSTGRES_PASSWORD=failover_test_only", "-e", "POSTGRES_DB=zocr_test",
               "postgres:16.6", "-c", "hba_file=/etc/zocr-pg_hba.conf", "-c", "wal_level=replica")
        created.append(primary)
        port = docker("port", primary, "5432/tcp").rsplit(":", 1)[1]
        repository = PostgresJobRepository(dsn=f"postgresql://postgres:failover_test_only@127.0.0.1:{port}/zocr_test")

        def ready_primary():
            with repository._connect(connect_timeout=2) as connection:
                return connection.execute("SELECT 1").fetchone()[0]

        wait_for(ready_primary)
        store = LocalJobStore(tmp_path / "store")
        repository.write_job("job", "default", {"status": "queued", "input": {"filename": "page.png"}})
        paths = store.job_paths("job")
        paths.input_dir.mkdir(parents=True)
        (paths.input_dir / "page.png").write_bytes(b"test image")
        docker("run", "-d", "--name", replica, *common, "--user", "postgres", "--entrypoint", "/bin/sh",
               "postgres:16.6", "-c",
               "pg_basebackup -d 'host=primary user=postgres password=failover_test_only' "
               "-D /var/lib/postgresql/replica -R -X stream && exec postgres -D /var/lib/postgresql/replica")
        created.append(replica)
        replica_port = docker("port", replica, "5432/tcp").rsplit(":", 1)[1]
        promoted = PostgresJobRepository(dsn=f"postgresql://postgres:failover_test_only@127.0.0.1:{replica_port}/zocr_test")
        with execution_session(store, repository, "job", "default") as old:
            old_job = repository.read_job("job", "default")
            old_job["status"] = "running"
            old.write_job(old_job)
            old_paths = result_paths(store, "job", old_job)
            old_paths.out_dir.mkdir(parents=True)
            with repository._connect() as connection:
                lsn = str(connection.execute("SELECT pg_current_wal_lsn()").fetchone()[0])

            def caught_up():
                with promoted._connect(connect_timeout=2) as connection:
                    return connection.execute("SELECT pg_is_in_recovery() AND pg_last_wal_replay_lsn() >= %s::pg_lsn",
                                              (lsn,)).fetchone()[0]

            wait_for(caught_up)
            docker("stop", primary)
            docker("exec", replica, "pg_ctl", "-D", "/var/lib/postgresql/replica", "promote", "-w")
            with pytest.raises(ExecutionLost):
                old.check()

            def runner(**kwargs):
                from pathlib import Path
                summary = {"inputs": kwargs["inputs"]}
                (Path(kwargs["outdir"]) / "pipeline_summary.json").write_text(json.dumps(summary), encoding="utf-8")
                return summary

            status, error, _ = process_job(job_id="job", tenant_id="default", job_store=store,
                                          job_repository=promoted, run_full_pipeline=runner)
            assert status == "succeeded", error
            published = promoted.read_job("job", "default")
            assert published["execution_token"] != old.token
            (old_paths.out_dir / "stale.txt").write_text("late old worker write", encoding="utf-8")
            with pytest.raises(ExecutionLost):
                old.write_job(dict(old_job, status="succeeded"))
            assert promoted.read_job("job", "default") == published
            assert not (result_paths(store, "job", published).out_dir / "stale.txt").exists()
    finally:
        for name in reversed(created):
            subprocess.run(["docker", "stop", name], capture_output=True, timeout=30)
