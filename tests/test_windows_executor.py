import os
import subprocess
import sys
import time

import pytest

from zocr.service.executor import execute_command

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Requires real Windows Job Objects")


def child_code(marker, ready):
    grandchild = f"import time,pathlib;time.sleep(4);pathlib.Path({str(marker)!r}).write_text('late')"
    return (
        "import subprocess,sys,pathlib,time;"
        f"subprocess.Popen([sys.executable,'-c',{grandchild!r}]);"
        f"pathlib.Path({str(ready)!r}).write_text('ready');time.sleep(30)"
    )


def test_windows_timeout_stops_descendants(tmp_path):
    marker, ready = tmp_path / "late.txt", tmp_path / "ready.txt"
    with pytest.raises(subprocess.TimeoutExpired):
        execute_command([sys.executable, "-c", child_code(marker, ready)], {}, 2)
    assert ready.exists()
    time.sleep(2.5)
    assert not marker.exists()


def test_windows_job_assignment_fails_closed_before_child_runs(tmp_path, monkeypatch):
    from zocr.service.windows_job import WindowsJob

    marker = tmp_path / "must-not-start.txt"

    def fail(self, process):
        raise OSError("Cannot establish job isolation")

    monkeypatch.setattr(WindowsJob, "assign_and_resume", fail)
    with pytest.raises(OSError, match="isolation"):
        execute_command([sys.executable, "-c", f"from pathlib import Path;Path({str(marker)!r}).write_text('bad')"], {}, 3)
    assert not marker.exists()


def test_windows_supervisor_death_closes_job_and_descendants(tmp_path):
    marker, ready = tmp_path / "late.txt", tmp_path / "ready.txt"
    supervisor = (
        "import sys;from zocr.service.executor import execute_command;"
        f"execute_command([sys.executable,'-c',{child_code(marker, ready)!r}],{{}},30)"
    )
    process = subprocess.Popen([sys.executable, "-c", supervisor])
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.05)
        assert ready.exists()
        process.kill()
        process.wait(timeout=5)
        time.sleep(4.5)
        assert not marker.exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
