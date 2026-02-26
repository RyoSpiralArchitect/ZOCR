from __future__ import annotations

import re
import subprocess
import sys


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_import_version() -> None:
    import zocr

    assert isinstance(zocr.__version__, str)
    assert re.match(r"^\d+\.\d+\.\d+$", zocr.__version__)


def test_zocr_help() -> None:
    proc = _run("-m", "zocr", "--help")
    assert proc.returncode == 0
    assert "python -m zocr" in proc.stdout


def test_zocr_version_flag() -> None:
    proc = _run("-m", "zocr", "--version")
    assert proc.returncode == 0
    assert proc.stdout.strip()


def test_diff_help() -> None:
    proc = _run("-m", "zocr.diff", "--help")
    assert proc.returncode == 0
    assert "semantic diff" in proc.stdout.lower()


def test_validate_help() -> None:
    proc = _run("-m", "zocr", "validate", "--help")
    assert proc.returncode == 0


def test_bench_help() -> None:
    proc = _run("-m", "zocr", "bench", "--help")
    assert proc.returncode == 0
