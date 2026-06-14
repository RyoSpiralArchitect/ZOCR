"""Quality gate reporting for contextual OCR exports."""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional

EXPORT_QUALITY_SCHEMA = "zocr.export_quality.v1"


def _env_float(name: str, default: float = 0.0) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except Exception:
        return default
    return value if math.isfinite(value) else default


def _bounded_env_float(
    name: str,
    default: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    value = _env_float(name, default)
    return float(max(minimum, min(maximum, value)))


def _quality_thresholds() -> Dict[str, float]:
    review_warn = _bounded_env_float("ZOCR_EXPORT_QUALITY_WARN_REVIEW_RATIO", 0.15)
    review_fail = max(
        review_warn,
        _bounded_env_float("ZOCR_EXPORT_QUALITY_FAIL_REVIEW_RATIO", 0.35),
    )
    low_conf_warn = _bounded_env_float("ZOCR_EXPORT_QUALITY_WARN_LOW_CONF_RATIO", 0.10)
    low_conf_fail = max(
        low_conf_warn,
        _bounded_env_float("ZOCR_EXPORT_QUALITY_FAIL_LOW_CONF_RATIO", 0.30),
    )
    surprisal_warn = _bounded_env_float("ZOCR_EXPORT_QUALITY_WARN_SURPRISAL_RATIO", 0.08)
    surprisal_fail = max(
        surprisal_warn,
        _bounded_env_float("ZOCR_EXPORT_QUALITY_FAIL_SURPRISAL_RATIO", 0.20),
    )
    coverage_warn = _bounded_env_float("ZOCR_EXPORT_QUALITY_WARN_COVERAGE_RATIO", 0.95)
    coverage_fail = min(
        coverage_warn,
        _bounded_env_float("ZOCR_EXPORT_QUALITY_FAIL_COVERAGE_RATIO", 0.50),
    )
    blank_warn = _bounded_env_float("ZOCR_EXPORT_QUALITY_WARN_BLANK_RATIO", 0.50)
    blank_fail = max(
        blank_warn,
        _bounded_env_float("ZOCR_EXPORT_QUALITY_FAIL_BLANK_RATIO", 0.95),
    )
    lexical_warn = _bounded_env_float(
        "ZOCR_EXPORT_QUALITY_WARN_LEXICAL_MEAN",
        0.55,
        maximum=1.6,
    )
    lexical_fail = min(
        lexical_warn,
        _bounded_env_float(
            "ZOCR_EXPORT_QUALITY_FAIL_LEXICAL_MEAN",
            0.35,
            maximum=1.6,
        ),
    )
    return {
        "review_warn": review_warn,
        "review_fail": review_fail,
        "low_conf_warn": low_conf_warn,
        "low_conf_fail": low_conf_fail,
        "surprisal_warn": surprisal_warn,
        "surprisal_fail": surprisal_fail,
        "coverage_warn": coverage_warn,
        "coverage_fail": coverage_fail,
        "blank_warn": blank_warn,
        "blank_fail": blank_fail,
        "lexical_warn": lexical_warn,
        "lexical_fail": lexical_fail,
    }


def _nested_count(payload: Dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, dict):
        raw = value.get("count")
    else:
        raw = value
    try:
        return int(raw or 0)
    except Exception:
        return 0


def _payload_float(payload: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(payload.get(key, default) or default)
    except Exception:
        value = default
    if not math.isfinite(value):
        return default
    return float(value)


def _status_above(value: float, warn_at: float, fail_at: float) -> str:
    if value >= fail_at:
        return "fail"
    if value >= warn_at:
        return "warn"
    return "pass"


def _status_below(value: float, warn_below: float, fail_below: float) -> str:
    if value < fail_below:
        return "fail"
    if value < warn_below:
        return "warn"
    return "pass"


def build_export_quality_report(
    export_stats: Dict[str, Any],
    signals_payload: Dict[str, Any],
    *,
    jsonl_path: str,
    signals_path: str,
    quality_path: str,
) -> Dict[str, Any]:
    """Build a machine-readable quality gate report for contextual OCR export."""

    thresholds = _quality_thresholds()
    records = _nested_count(export_stats, "records")
    cells_total = _nested_count(export_stats, "cells_total")
    pages = _nested_count(export_stats, "pages")
    tables = _nested_count(export_stats, "tables")
    low_conf_ratio = _payload_float(signals_payload, "low_conf_ratio")
    high_surprisal_ratio = _payload_float(signals_payload, "high_surprisal_ratio")
    review_ratio = _payload_float(signals_payload, "review_ratio")
    missing_pages = _nested_count(export_stats, "missing_pages")
    guard_meta = export_stats.get("guard") if isinstance(export_stats.get("guard"), dict) else {}
    guard_timeouts = _nested_count(guard_meta, "timeouts")
    blank_meta = export_stats.get("blank_skip") if isinstance(export_stats.get("blank_skip"), dict) else {}
    blank_ratio = _payload_float(blank_meta, "ratio")
    lexical_meta = export_stats.get("lexical_quality")
    lexical_mean: Optional[float] = None
    if isinstance(lexical_meta, dict) and lexical_meta.get("samples"):
        lexical_mean = _payload_float(lexical_meta, "mean")
    max_cells = _nested_count(export_stats, "max_cells")
    truncated = bool(export_stats.get("truncated"))
    coverage_ratio = float(records / float(max(1, cells_total))) if cells_total > 0 else 1.0

    gates: List[Dict[str, Any]] = []

    def _add_gate(name: str, status: str, value: Any, message: str, **extra: Any) -> None:
        gate: Dict[str, Any] = {
            "name": name,
            "status": status,
            "value": value,
            "message": message,
        }
        for key, extra_value in extra.items():
            gate[key] = extra_value
        gates.append(gate)

    if cells_total > 0 or tables > 0:
        record_status = "pass" if records > 0 else "fail"
        _add_gate(
            "records_present",
            record_status,
            records,
            "Contextual records were written for discovered table cells."
            if record_status == "pass"
            else "Tables or cells were discovered but no contextual records were written.",
            minimum=1,
        )
    else:
        _add_gate(
            "records_present",
            "pass",
            records,
            "No table cells were discovered, so an empty export is expected.",
            minimum=0,
        )

    coverage_status = (
        _status_below(
            coverage_ratio,
            thresholds["coverage_warn"],
            thresholds["coverage_fail"],
        )
        if cells_total > 0
        else "pass"
    )
    _add_gate(
        "record_coverage",
        coverage_status,
        round(coverage_ratio, 4),
        "Written records cover the cells visited during export.",
        warn_below=thresholds["coverage_warn"],
        fail_below=thresholds["coverage_fail"],
    )

    low_conf_status = _status_above(
        low_conf_ratio,
        thresholds["low_conf_warn"],
        thresholds["low_conf_fail"],
    )
    _add_gate(
        "low_conf_ratio",
        low_conf_status,
        round(low_conf_ratio, 4),
        "Share of records below the configured OCR confidence threshold.",
        warn_at=thresholds["low_conf_warn"],
        fail_at=thresholds["low_conf_fail"],
    )

    review_status = _status_above(
        review_ratio,
        thresholds["review_warn"],
        thresholds["review_fail"],
    )
    _add_gate(
        "review_ratio",
        review_status,
        round(review_ratio, 4),
        "Share of records queued for learning or human review.",
        warn_at=thresholds["review_warn"],
        fail_at=thresholds["review_fail"],
    )

    surprisal_status = _status_above(
        high_surprisal_ratio,
        thresholds["surprisal_warn"],
        thresholds["surprisal_fail"],
    )
    _add_gate(
        "high_surprisal_ratio",
        surprisal_status,
        round(high_surprisal_ratio, 4),
        "Share of records flagged as contextually unlikely by the N-gram model.",
        warn_at=thresholds["surprisal_warn"],
        fail_at=thresholds["surprisal_fail"],
    )

    if lexical_mean is not None:
        lexical_status = _status_below(
            lexical_mean,
            thresholds["lexical_warn"],
            thresholds["lexical_fail"],
        )
        _add_gate(
            "lexical_quality_mean",
            lexical_status,
            round(float(lexical_mean), 4),
            "Mean lexical quality over OCR text candidates.",
            warn_below=thresholds["lexical_warn"],
            fail_below=thresholds["lexical_fail"],
        )

    if blank_meta:
        blank_status = _status_above(
            blank_ratio,
            thresholds["blank_warn"],
            thresholds["blank_fail"],
        )
        _add_gate(
            "blank_skip_ratio",
            blank_status,
            round(blank_ratio, 4),
            "Share of visited cells skipped by blank-crop detection.",
            warn_at=thresholds["blank_warn"],
            fail_at=thresholds["blank_fail"],
        )

    missing_status = "pass"
    if missing_pages:
        missing_status = "fail" if records == 0 and pages > 0 else "warn"
    _add_gate(
        "page_bitmap_availability",
        missing_status,
        missing_pages,
        "All referenced page bitmaps were available."
        if missing_status == "pass"
        else "Some referenced page bitmaps could not be opened.",
    )

    guard_status = "pass"
    if guard_timeouts:
        guard_status = "fail" if records == 0 else "warn"
    _add_gate(
        "guard_timeouts",
        guard_status,
        guard_timeouts,
        "No per-table export guard timeouts occurred."
        if guard_status == "pass"
        else "One or more tables hit the export guard deadline.",
    )

    trunc_status = "pass"
    if truncated:
        trunc_status = "fail" if records == 0 else "warn"
    _add_gate(
        "cell_limit",
        trunc_status,
        {"max_cells": max_cells, "truncated": truncated},
        "The export completed without hitting ZOCR_EXPORT_MAX_CELLS."
        if trunc_status == "pass"
        else "The export stopped early because ZOCR_EXPORT_MAX_CELLS was reached.",
    )

    status_order = {"pass": 0, "warn": 1, "fail": 2}
    overall_status = "pass"
    for gate in gates:
        gate_status = str(gate.get("status") or "pass")
        if status_order.get(gate_status, 0) > status_order.get(overall_status, 0):
            overall_status = gate_status

    score = 100.0
    for gate in gates:
        if gate.get("status") == "fail":
            score -= 25.0
        elif gate.get("status") == "warn":
            score -= 7.5
    score = round(max(0.0, min(100.0, score)), 1)

    recommendations: List[str] = []
    problem_names = {str(gate.get("name")) for gate in gates if gate.get("status") in {"warn", "fail"}}
    if "page_bitmap_availability" in problem_names:
        recommendations.append("Verify the source image mapping, page image paths, or PDF rasterization output.")
    if {"low_conf_ratio", "review_ratio"} & problem_names:
        recommendations.append("Inspect the paired learning JSONL or run the reanalysis pass before indexing.")
    if "high_surprisal_ratio" in problem_names:
        recommendations.append("Check domain/profile selection and review high-surprisal cells for OCR drift.")
    if "lexical_quality_mean" in problem_names:
        recommendations.append("Tune the toy OCR policy or rerun with a stronger OCR backend for noisy text.")
    if "guard_timeouts" in problem_names:
        recommendations.append("Raise ZOCR_EXPORT_GUARD_MS or disable the guard for long tables.")
    if "cell_limit" in problem_names:
        recommendations.append("Increase ZOCR_EXPORT_MAX_CELLS when a complete export is required.")
    if "blank_skip_ratio" in problem_names:
        recommendations.append("Confirm the page crop geometry; a high blank ratio often means table bounds are misaligned.")

    return {
        "schema": EXPORT_QUALITY_SCHEMA,
        "status": overall_status,
        "score": score,
        "summary": {
            "records": records,
            "cells_total": cells_total,
            "coverage_ratio": round(coverage_ratio, 4),
            "pages": pages,
            "tables": tables,
            "low_conf_ratio": round(low_conf_ratio, 4),
            "high_surprisal_ratio": round(high_surprisal_ratio, 4),
            "review_ratio": round(review_ratio, 4),
            "missing_pages": missing_pages,
            "guard_timeouts": guard_timeouts,
            "blank_skip_ratio": round(blank_ratio, 4) if blank_meta else None,
            "lexical_quality_mean": round(float(lexical_mean), 4) if lexical_mean is not None else None,
            "truncated": truncated,
        },
        "thresholds": thresholds,
        "gates": gates,
        "recommendations": recommendations,
        "artifacts": {
            "jsonl": jsonl_path,
            "signals_json": signals_path,
            "quality_json": quality_path,
            "learning_jsonl": signals_payload.get("learning_jsonl"),
        },
    }


__all__ = ["EXPORT_QUALITY_SCHEMA", "build_export_quality_report"]
