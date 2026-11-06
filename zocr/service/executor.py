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


def execute_command(command: list[str], payload: dict[str, Any], timeout_sec: float) -> dict[str, Any]:
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        text=True, start_new_session=os.name == "posix",
    )
    try:
        output, _ = process.communicate(
            json.dumps(payload), timeout=timeout_sec if timeout_sec > 0 else None,
        )
    except BaseException:
        if os.name == "posix":
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.communicate()
        raise
    if process.returncode:
        raise RuntimeError(f"Pipeline process failed (exit {process.returncode})")
    result = json.loads(output)
    if not isinstance(result, dict):
        raise RuntimeError("Pipeline process returned an invalid summary")
    return result


def execute_pipeline(*, timeout_sec: float = 900.0, **kwargs: Any) -> dict[str, Any]:
    try:
        return execute_command(
            [sys.executable, "-m", "zocr.service.executor", str(os.getpid())], kwargs, timeout_sec,
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
