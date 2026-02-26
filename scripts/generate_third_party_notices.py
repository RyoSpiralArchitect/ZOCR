#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from importlib import metadata
from pathlib import Path
from typing import Iterable, Optional


def _norm_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", (name or "").strip()).lower()


def _first_nonempty(values: Iterable[Optional[str]]) -> Optional[str]:
    for value in values:
        if not value:
            continue
        stripped = value.strip()
        if stripped:
            return stripped
    return None


def _pick_url(meta: "metadata.PackageMetadata") -> str:
    url = _first_nonempty(
        [
            meta.get("Home-page"),
            meta.get("Home-Page"),
            meta.get("Homepage"),
        ]
    )
    if url:
        return url

    project_urls: list[str] = []
    try:
        project_urls = list(meta.get_all("Project-URL") or [])
    except Exception:
        project_urls = []

    parsed: dict[str, str] = {}
    for raw in project_urls:
        raw = (raw or "").strip()
        if not raw:
            continue
        label, sep, value = raw.partition(",")
        if not sep:
            continue
        label = label.strip().lower()
        value = value.strip()
        if label and value:
            parsed[label] = value

    for preferred in ("homepage", "home", "source", "repository", "documentation", "changelog"):
        if preferred in parsed:
            return parsed[preferred]

    if parsed:
        return parsed[sorted(parsed.keys())[0]]
    return ""


def _pick_license(meta: "metadata.PackageMetadata") -> str:
    raw = (meta.get("License") or "").strip()
    if raw and raw.upper() not in {"UNKNOWN", "UNLICENSED"}:
        return raw

    classifiers: list[str] = []
    try:
        classifiers = list(meta.get_all("Classifier") or [])
    except Exception:
        classifiers = []

    for classifier in classifiers:
        classifier = (classifier or "").strip()
        if classifier.startswith("License ::"):
            return classifier.split("::")[-1].strip()
    return "UNKNOWN"


def _iter_rows(*, exclude_norm_names: set[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for dist in metadata.distributions():
        try:
            meta = dist.metadata
        except Exception:
            continue
        name = (meta.get("Name") or getattr(dist, "name", "") or "").strip()
        if not name:
            continue
        if _norm_name(name) in exclude_norm_names:
            continue
        rows.append(
            {
                "name": name,
                "version": (getattr(dist, "version", "") or "").strip(),
                "license": _pick_license(meta),
                "url": _pick_url(meta),
            }
        )

    rows.sort(key=lambda r: _norm_name(r["name"]))
    return rows


def _render_markdown(rows: list[dict[str, str]], *, generated_at: str) -> str:
    lines: list[str] = []
    lines.append("# Third-Party Notices")
    lines.append("")
    lines.append(
        "This file lists third-party dependencies detected in the Python environment used to generate it."
    )
    lines.append(
        "It is generated from package metadata (which may be incomplete). For authoritative terms, consult each package’s"
        " LICENSE/COPYING files and its upstream project page."
    )
    lines.append("")
    lines.append(f"Generated at: `{generated_at}`")
    lines.append("")
    lines.append("## Python packages")
    lines.append("")
    lines.append("| Package | Version | License | URL |")
    lines.append("|---|---:|---|---|")
    for row in rows:
        name = row["name"].replace("|", "\\|")
        version = row["version"].replace("|", "\\|")
        license_name = row["license"].replace("|", "\\|")
        url = row["url"].replace("|", "\\|")
        lines.append(f"| {name} | {version} | {license_name} | {url} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser("generate_third_party_notices")
    parser.add_argument(
        "-o",
        "--output",
        default="-",
        help="Output path (use '-' for stdout).",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Normalized package name to exclude (repeatable).",
    )
    args = parser.parse_args(argv)

    exclude_norm_names = {_norm_name("zocr-suite")}
    for raw in args.exclude:
        exclude_norm_names.add(_norm_name(raw))

    rows = _iter_rows(exclude_norm_names=exclude_norm_names)
    generated_at = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    text = _render_markdown(rows, generated_at=generated_at)

    if args.output == "-":
        sys.stdout.write(text)
        return
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    main()
