import json
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from zocr.orchestrator.resume import prepare_resume, reusable_steps, step_integrity
from zocr.service.execution_lock import execution_lock
from zocr.service.executor import execute_command
from zocr.service.jobs import process_job
from zocr.service.storage import LocalJobStore


def test_executor_isolates_environment(monkeypatch):
    monkeypatch.delenv("ZOCR_CHILD_TEST", raising=False)
    result = execute_command([sys.executable, "-c", (
        "import json,os; os.environ['ZOCR_CHILD_TEST']='child'; "
        "print(json.dumps({'pid':os.getpid()}))"
    )], {}, 5)
    import os
    assert result["pid"] != os.getpid()
    assert "ZOCR_CHILD_TEST" not in os.environ


def test_executor_stops_writes_after_timeout(tmp_path):
    marker = tmp_path / "late.txt"
    program = (
        "import time,pathlib; time.sleep(.4); "
        f"pathlib.Path({str(marker)!r}).write_text('late')"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        execute_command([sys.executable, "-c", program], {}, .05)
    time.sleep(.5)
    assert not marker.exists()


def test_executor_kills_descendants_on_timeout(tmp_path):
    marker = tmp_path / "descendant.txt"
    child = f"import time,pathlib;time.sleep(.4);pathlib.Path({str(marker)!r}).write_text('late')"
    parent = f"import subprocess,sys,time;subprocess.Popen([sys.executable,'-c',{child!r}]);time.sleep(10)"
    with pytest.raises(subprocess.TimeoutExpired):
        execute_command([sys.executable, "-c", parent], {}, .1)
    time.sleep(.5)
    assert not marker.exists()


def test_execution_lock_excludes_duplicates_and_releases(tmp_path):
    store = LocalJobStore(tmp_path)
    repository = SimpleNamespace(backend="local")
    with execution_lock(store, repository, "job", "tenant") as first:
        assert first
        with execution_lock(store, repository, "job", "tenant") as second:
            assert not second
        with execution_lock(store, repository, "other", "tenant") as other:
            assert other
    with execution_lock(store, repository, "job", "tenant") as again:
        assert again


@pytest.mark.parametrize("status", ["succeeded", "failed"])
def test_terminal_job_is_not_executed_again(tmp_path, status):
    store = LocalJobStore(tmp_path)
    repository = MagicMock()
    repository.backend = "local"
    repository.read_job.return_value = {"status": status}
    runner = MagicMock()
    result = process_job(job_id="job", tenant_id="tenant", job_store=store,
                         job_repository=repository, run_full_pipeline=runner)
    assert result[0] == status
    runner.assert_not_called()
    repository.write_job.assert_not_called()


def test_resume_rejects_changed_options_input_and_outputs(tmp_path):
    image = tmp_path / "page.png"
    image.write_bytes(b"first")
    out = tmp_path / "out"
    out.mkdir()
    options = {"inputs": [str(image)], "dpi": 200}
    signature = prepare_resume(str(out), options)
    assert signature == prepare_resume(str(out), {**options, "resume": True})
    assert signature != prepare_resume(str(out), {**options, "dpi": 300})
    artifact = out / "doc.zocr.json"
    artifact.write_text('{}', encoding="utf-8")
    history = out / "pipeline_history.jsonl"
    record = {"name": "OCR", "ok": True, "run_fingerprint": signature,
              "output_hashes": step_integrity(str(out), "OCR")}
    history.write_text(json.dumps(record) + "\n", encoding="utf-8")
    assert reusable_steps(str(out), signature) == {"OCR"}
    artifact.write_text('{"corrupt":true}', encoding="utf-8")
    assert reusable_steps(str(out), signature) == set()
    image.write_bytes(b"second")
    assert signature != prepare_resume(str(out), options)


def test_queue_heartbeat_refreshes_pending_message():
    from threading import Event
    from zocr.service.queue import RedisStreamQueue

    queue = RedisStreamQueue.__new__(RedisStreamQueue)
    queue.stream = "jobs"
    queue._client = MagicMock()
    called = Event()
    queue._client.xclaim.side_effect = lambda *args, **kwargs: called.set()
    with queue.keep_alive(group="workers", consumer="one", message_id="1-0", interval=.01):
        assert called.wait(2)
    count = queue._client.xclaim.call_count
    time.sleep(.03)
    assert queue._client.xclaim.call_count == count
    queue._client.xclaim.assert_called_with("jobs", "workers", "one", 0, ["1-0"], justid=True)


def test_postgres_execution_lock_releases_session_lock():
    repository = MagicMock()
    repository.backend = "postgres"
    cursor = repository._connect.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value
    cursor.fetchone.return_value = (True,)
    with pytest.raises(RuntimeError, match="runner failed"):
        with execution_lock(None, repository, "job", "tenant") as acquired:
            assert acquired
            raise RuntimeError("runner failed")
    assert cursor.execute.call_count == 2
    assert "pg_advisory_unlock" in cursor.execute.call_args.args[0]


def test_resume_latest_failure_invalidates_success(tmp_path):
    artifact = tmp_path / "doc.zocr.json"
    artifact.write_text('{}', encoding="utf-8")
    records = [
        {"name": "OCR", "ok": True, "run_fingerprint": "run",
         "output_hashes": step_integrity(str(tmp_path), "OCR")},
        {"name": "OCR", "ok": False, "run_fingerprint": "run"},
    ]
    (tmp_path / "pipeline_history.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8",
    )
    assert reusable_steps(str(tmp_path), "run") == set()
