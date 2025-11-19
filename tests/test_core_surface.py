"""Regression tests for the lazy zocr.core surface."""

from __future__ import annotations

import importlib


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
