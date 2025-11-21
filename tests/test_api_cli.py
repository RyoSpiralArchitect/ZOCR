# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024 ZOCR contributors

import json

from zocr import api_cli


def test_api_cli_ingest_dry_run(tmp_path, capsys):
    files = [str(tmp_path / "sample.pdf")]
    api_cli.main(
        [
            "ingest",
            "--tenant",
            "tenantA",
            "--job-id",
            "episode_cli",
            "--out-root",
            str(tmp_path),
            "--dry-run",
            *files,
        ]
    )

    captured = capsys.readouterr().out
    payload = json.loads(captured)
    assert payload["job_id"] == "episode_cli"
    assert payload["status"] == "queued"
    assert payload["tenant_id"] == "tenantA"
    assert payload["artifacts"]["rag_manifest"].endswith("manifest.json")


def test_api_cli_query_reads_manifest(tmp_path, capsys):
    job_id = "episode_cli_query"
    job_dir = tmp_path / job_id / "rag"
    job_dir.mkdir(parents=True)
    manifest = {
        "cells": {"count": 1},
        "page_sections": 1,
        "table_sections": 1,
        "trace_schema": {"label": "doc=demo;page=1;row=1"},
    }
    manifest_path = job_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    api_cli.main(
        [
            "query",
            "--tenant",
            "tenantB",
            "--job-id",
            job_id,
            "--base-dir",
            str(tmp_path),
            "--query",
            "売上推移は?",
        ]
    )

    captured = capsys.readouterr().out
    payload = json.loads(captured)
    assert payload["status"] == "ready"
    assert payload["artifacts"]["sources"]["manifest"] == str(manifest_path)
    assert payload["answer"]["business_commentary"].startswith("Received query")
    assert payload["provenance"][0]["trace"].startswith("doc=")
