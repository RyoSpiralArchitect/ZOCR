"""Regression tests for the lazy zocr.core surface."""

from __future__ import annotations

import importlib
import subprocess
import sys


def test_core_surface_exposes_compat_module() -> None:
    import zocr.core as core

    compat = importlib.import_module("zocr.core.zocr_core")
    assert core.zocr_core is compat


def test_core_surface_proxies_to_split_modules() -> None:
    import zocr.core as core

    from zocr.core import augmenter, indexer, query_engine, exporters, monitoring

    assert core.augment is augmenter.augment
    assert core.build_index is indexer.build_index
    assert core.query is query_engine.query
    assert core.hybrid_query is query_engine.hybrid_query
    assert core.sql_export is exporters.sql_export
    assert core.export_rag_bundle is exporters.export_rag_bundle
    assert core.monitor is monitoring.monitor
    assert core.learn_from_monitor is monitoring.learn_from_monitor
    assert core.autotune_unlabeled is monitoring.autotune_unlabeled
    assert (
        core.metric_col_over_under_rate is monitoring.metric_col_over_under_rate
    )
    assert core.metric_chunk_consistency is monitoring.metric_chunk_consistency
    assert (
        core.metric_col_alignment_energy_cached
        is monitoring.metric_col_alignment_energy_cached
    )


def test_core_surface_exposes_split_modules() -> None:
    import zocr.core as core

    assert core.augmenter is importlib.import_module("zocr.core.augmenter")
    assert core.base is importlib.import_module("zocr.core.base")
    assert core.domains is importlib.import_module("zocr.core.domains")
    assert core.exporters is importlib.import_module("zocr.core.exporters")
    assert core.indexer is importlib.import_module("zocr.core.indexer")
    assert core.monitoring is importlib.import_module("zocr.core.monitoring")
    assert core.numba_support is importlib.import_module("zocr.core.numba_support")
    assert core.query_engine is importlib.import_module("zocr.core.query_engine")
    assert core.tokenization is importlib.import_module("zocr.core.tokenization")


def test_core_surface_all_stays_in_sync_with_exports() -> None:
    import zocr.core as core

    expected = {
        "zocr_core",
        "augment",
        "build_index",
        "query",
        "hybrid_query",
        "embed_jsonl",
        "sql_export",
        "export_rag_bundle",
        "monitor",
        "learn_from_monitor",
        "autotune_unlabeled",
        "metric_col_over_under_rate",
        "metric_chunk_consistency",
        "metric_col_alignment_energy_cached",
        "main",
        "augmenter",
        "base",
        "domains",
        "embedders",
        "exporters",
        "indexer",
        "monitoring",
        "numba_support",
        "query_engine",
        "tokenization",
    }

    assert expected == set(core.__all__)


def test_core_package_can_run_as_module() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "zocr.core", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "ZOCR Multi-domain Core" in result.stdout
