import json
from pathlib import Path

import pytest

from zocr.service.jobs import process_job
from zocr.service.metadata import LocalJobRepository
from zocr.service.storage import LocalJobStore


def test_execution_writes_only_to_unique_attempts(tmp_path):
    store = LocalJobStore(tmp_path)
    repository = LocalJobRepository(job_store=store)
    repository.write_job("job", "default", {"status": "queued", "input": {"filename": "page.png"}})
    paths = store.job_paths("job")
    paths.input_dir.mkdir()
    (paths.input_dir / "page.png").write_bytes(b"image")
    seen = []

    def runner(**kwargs):
        out = Path(kwargs["outdir"])
        seen.append(out)
        summary = {"inputs": kwargs["inputs"]}
        (out / "pipeline_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return summary

    status, error, _ = process_job(job_id="job", tenant_id="default", job_store=store,
                                  job_repository=repository, run_full_pipeline=runner)
    assert status == "succeeded", error
    job = repository.read_job("job", "default")
    assert seen[0] == paths.root / "attempts" / job["execution_token"] / "out"
    assert not paths.out_dir.exists()
    assert job["artifacts"]["out_dir"] == f"attempts/{job['execution_token']}/out"


def test_lost_execution_session_cannot_publish(tmp_path):
    from zocr.service.execution_lock import ExecutionLost, execution_session

    store = LocalJobStore(tmp_path)
    repository = LocalJobRepository(job_store=store)
    repository.write_job("job", "default", {"status": "queued"})
    with execution_session(store, repository, "job", "default") as guard:
        job = repository.read_job("job", "default")
        guard.write_job(job)
        replacement = dict(job, execution_token="new-owner")
        repository.write_job("job", "default", replacement)
        with pytest.raises(ExecutionLost):
            guard.write_job(dict(job, status="succeeded"))
    assert repository.read_job("job", "default")["execution_token"] == "new-owner"
