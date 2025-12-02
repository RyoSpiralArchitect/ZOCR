"""Evaluate visual-heavy tech docs benchmark.

Given a predictions JSON file, this script measures:
- Top-k visual hit rate for diagram/table questions (expected visual targets vs. retrieved visuals).
- End-to-end answer accuracy for questions whose answers require reading a visual element.

The benchmark definition lives in ``tests/fixtures/techdocs/benchmark.json`` and is intentionally
small so it can ship with the repository. See the README in the fixtures folder for guidance on
adding new PDFs or questions.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


@dataclass
class BenchmarkExample:
    id: str
    pdf: str
    page: int
    query: str
    answers: Sequence[str]
    visual_type: str
    visual_target: str
    answer_requires_visual: bool


@dataclass
class Prediction:
    id: str
    predicted_answer: str | None
    retrieved_visuals: List[str]


@dataclass
class Metrics:
    diagram_table_hits: int = 0
    diagram_table_total: int = 0
    visual_answer_hits: int = 0
    visual_answer_total: int = 0
    missing_predictions: int = 0

    def diagram_table_rate(self) -> float:
        return self.diagram_table_hits / self.diagram_table_total if self.diagram_table_total else 0.0

    def visual_answer_accuracy(self) -> float:
        return self.visual_answer_hits / self.visual_answer_total if self.visual_answer_total else 0.0


NORMALIZE_RE = re.compile(r"\s+")


def _normalize(text: str | None) -> str:
    if text is None:
        return ""
    return NORMALIZE_RE.sub(" ", text).strip().lower()


def load_benchmark(path: Path) -> list[BenchmarkExample]:
    data = json.loads(path.read_text())
    examples: list[BenchmarkExample] = []
    for entry in data:
        examples.append(
            BenchmarkExample(
                id=entry["id"],
                pdf=entry["pdf"],
                page=entry.get("page", 1),
                query=entry["query"],
                answers=entry.get("answers", []),
                visual_type=entry.get("visual_type", ""),
                visual_target=entry.get("visual_target", ""),
                answer_requires_visual=bool(entry.get("answer_requires_visual", False)),
            )
        )
    return examples


def load_predictions(path: Path) -> dict[str, Prediction]:
    data = json.loads(path.read_text())
    predictions: dict[str, Prediction] = {}
    for entry in data:
        retrieved = entry.get("retrieved_visuals", [])
        if isinstance(retrieved, str):
            retrieved_visuals = [retrieved]
        else:
            retrieved_visuals = list(retrieved)
        predictions[entry["id"]] = Prediction(
            id=entry["id"],
            predicted_answer=entry.get("predicted_answer"),
            retrieved_visuals=retrieved_visuals,
        )
    return predictions


def _answer_is_correct(predicted: str | None, valid_answers: Sequence[str]) -> bool:
    if predicted is None:
        return False
    norm_pred = _normalize(predicted)
    return any(_normalize(ans) == norm_pred for ans in valid_answers)


def _visual_hit(expected: str, retrieved_visuals: Sequence[str], top_k: int) -> bool:
    expected_norm = _normalize(expected)
    limited = list(retrieved_visuals)[:top_k]
    return any(expected_norm in _normalize(item) for item in limited)


def evaluate(examples: Iterable[BenchmarkExample], predictions: dict[str, Prediction], top_k: int) -> Metrics:
    metrics = Metrics()
    for example in examples:
        pred = predictions.get(example.id)
        if example.visual_type in {"diagram", "table"} and example.visual_target:
            metrics.diagram_table_total += 1
            retrieved = pred.retrieved_visuals if pred else []
            if _visual_hit(example.visual_target, retrieved, top_k):
                metrics.diagram_table_hits += 1

        if example.answer_requires_visual:
            metrics.visual_answer_total += 1
            if pred is None:
                metrics.missing_predictions += 1
            if _answer_is_correct(pred.predicted_answer if pred else None, example.answers):
                metrics.visual_answer_hits += 1
    return metrics


def format_report(metrics: Metrics, top_k: int) -> str:
    lines = [
        f"Top-{top_k} visual hit rate (diagram/table): {metrics.diagram_table_hits}/{metrics.diagram_table_total} = {metrics.diagram_table_rate():.2f}",
        f"Answer accuracy when visuals are required: {metrics.visual_answer_hits}/{metrics.visual_answer_total} = {metrics.visual_answer_accuracy():.2f}",
        f"Missing predictions: {metrics.missing_predictions}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate tech docs visual benchmark.")
    parser.add_argument(
        "--predictions",
        required=True,
        type=Path,
        help="Path to a JSON file containing model predictions (list of objects with id, predicted_answer, retrieved_visuals).",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("tests/fixtures/techdocs/benchmark.json"),
        help="Path to the benchmark JSON file (defaults to bundled techdocs benchmark).",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Top-k cutoff for visual hit-rate calculation.")
    args = parser.parse_args()

    examples = load_benchmark(args.benchmark)
    predictions = load_predictions(args.predictions)

    metrics = evaluate(examples, predictions, args.top_k)
    print(format_report(metrics, args.top_k))


if __name__ == "__main__":
    main()
