from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from statistics import median
from typing import Any, Dict, List, Optional
from random import betavariate


@dataclass
class PriorDecision:
    """Lightweight record describing a motion-prior decision."""

    use_prior: bool
    sigma_px: Optional[float]
    reason: str = ""


class PriorBandit:
    """Two-armed Thompson Sampling bandit for toggling the motion prior."""

    def __init__(self, state_path: str):
        self.state_path = state_path
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {}
        except Exception:
            data = {}
        self.state: Dict[str, Dict[str, Dict[str, float]]] = data

    def _get(self, signature: str, action: str) -> Dict[str, float]:
        self.state.setdefault(signature, {})
        bucket = self.state[signature]
        bucket.setdefault(action, {"a": 1.0, "b": 1.0})
        return bucket[action]

    def decide(self, signature: str) -> str:
        weights: Dict[str, float] = {}
        for action in ("WITH_PRIOR", "NO_PRIOR"):
            ab = self._get(signature, action)
            weights[action] = betavariate(max(1e-3, ab.get("a", 1.0)), max(1e-3, ab.get("b", 1.0)))
        return max(weights, key=weights.get)

    def update(self, signature: str, action: str, success: bool) -> None:
        ab = self._get(signature, action)
        key = "a" if success else "b"
        ab[key] = float(ab.get(key, 1.0) + 1.0)

    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        except Exception:
            pass
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)


def normalize_headers_to_signature(headers: Optional[List[str]]) -> str:
    tokens: List[str] = []
    if headers:
        for header in headers:
            if not isinstance(header, str):
                continue
            clean = "".join(ch for ch in header.lower() if ch.isalnum())
            if clean:
                tokens.append(clean)
    if not tokens:
        return "unknown"
    tokens.sort()
    joined = "|".join(tokens)
    return hashlib.md5(joined.encode("utf-8")).hexdigest()[:12]


def estimate_sigma_px(
    delta_y: List[float],
    median_row_h: float,
    s_min_ratio: float = 0.15,
    s_max_ratio: float = 1.5,
) -> float:
    if not delta_y:
        return max(1.0, 0.5 * float(median_row_h or 1.0))
    med = median(delta_y)
    mad = median([abs(d - med) for d in delta_y])
    sigma = 1.4826 * mad
    sigma_min = s_min_ratio * max(1.0, median_row_h)
    sigma_max = s_max_ratio * max(1.0, median_row_h)
    return float(min(max(sigma, sigma_min), sigma_max))


def decide_success(summary: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(summary, dict):
        return False
    monitor = summary.get("monitor_row") or {}
    gate = monitor.get("gate_pass") if isinstance(monitor, dict) else None
    if gate is not True:
        return False
    metrics = summary.get("consensus_metrics") or {}
    agg = metrics.get("aggregate") if isinstance(metrics, dict) else None
    if isinstance(agg, dict):
        teds = agg.get("teds_mean")
        if isinstance(teds, (int, float)) and teds < 0.75:
            return False
        outlier = agg.get("row_outlier_rate_med")
        if isinstance(outlier, (int, float)) and outlier > 0.25:
            return False
    return True
