from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
from threading import Thread
import time
from typing import Any


def execute_command(command: list[str], payload: dict[str, Any], timeout_sec: float,
                    ownership_check=None) -> dict[str, Any]:
    job = None
    if os.name == "nt":
        from .windows_job import WindowsJob
        job = WindowsJob()
    try:
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            start_new_session=os.name == "posix",
            creationflags=0x00000004 if job is not None else 0,
        )
    except BaseException:
        if job is not None:
            job.close()
        raise
    try:
        if job is not None:
            job.assign_and_resume(process)
        deadline = time.monotonic() + timeout_sec if timeout_sec > 0 else None
        data = json.dumps(payload)
        while True:
            if ownership_check is not None:
                ownership_check()
            remaining = max(.001, deadline - time.monotonic()) if deadline is not None else None
            interval = min(.5, remaining) if ownership_check is not None and remaining is not None else remaining
            if ownership_check is not None and interval is None:
                interval = .5
            try:
                output, _ = process.communicate(data, timeout=interval)
                break
            except subprocess.TimeoutExpired:
                data = None
                if ownership_check is None or (deadline is not None and time.monotonic() >= deadline):
                    raise
    except BaseException:
        if os.name == "posix":
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        else:
            if job is not None:
                job.close()
            with contextlib.suppress(OSError):
                process.kill()
        process.communicate()
        raise
    finally:
        if job is not None:
            job.close()
    if process.returncode:
        raise RuntimeError(f"Pipeline process failed (exit {process.returncode})")
    result = json.loads(output)
    if not isinstance(result, dict):
        raise RuntimeError("Pipeline process returned an invalid summary")
    return result


def execute_pipeline(*, timeout_sec: float = 900.0, ownership_check=None, **kwargs: Any) -> dict[str, Any]:
    try:
        return execute_command(
            [sys.executable, "-m", "zocr.service.executor", str(os.getpid())], kwargs, timeout_sec,
            ownership_check=ownership_check,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError("Pipeline timeout") from exc


def watch_parent(parent_pid: int) -> None:
    if os.name != "posix":
        return

    def monitor():
        while os.getppid() == parent_pid:
            time.sleep(.1)
        if os.getpgrp() == os.getpid():
            os.killpg(os.getpid(), signal.SIGKILL)
        os._exit(1)

    Thread(target=monitor, daemon=True).start()


def main() -> None:
    if len(sys.argv) > 1:
        watch_parent(int(sys.argv[1]))
    payload = json.load(sys.stdin)
    with contextlib.redirect_stdout(sys.stderr):
        from zocr.orchestrator.zocr_pipeline import run_full_pipeline
        from zocr.utils.json_utils import json_ready

        result = json_ready(run_full_pipeline(**payload))
    json.dump(result, sys.stdout, ensure_ascii=False, allow_nan=False)


if __name__ == "__main__":
    main()
