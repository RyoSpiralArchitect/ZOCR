import hashlib
import json

import pytest
from PIL import Image, ImageDraw

from zocr.consensus.cell_evaluation import edit_distance, evaluate_cells, load_cell_dataset, text_metrics


def cell_dataset(tmp_path):
    image = Image.new("L", (100, 32), 255)
    ImageDraw.Draw(image).text((4, 5), "2358", fill=0)
    image.save(tmp_path / "cell.png")
    row = {"id": "cell-1", "image": "cell.png", "document": "cell.png", "group_id": "test-doc",
           "text": "2358", "split": "test", "source_kind": "fixture", "approved": True,
           "reviewer": "fixture-owner", "reason": "Known rendered text"}
    path = tmp_path / "cells.jsonl"
    path.write_text(json.dumps(row), encoding="utf-8")
    return path, row


def test_raw_cer_wer_preserve_blank_errors_and_unicode():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("猫", "犬") == 1
    metrics = text_metrics([("cat dog", "cat"), ("", "ghost")])
    assert metrics["character_edits"] == 9
    assert metrics["cer"] == 9 / 7
    assert metrics["word_edits"] == 2
    blank = text_metrics([("", "")])
    assert blank["cer"] is None and blank["exact_match_rate"] == 1
    assert text_metrics([("", "x")])["character_edits"] == 1
    with pytest.raises(ValueError):
        text_metrics([])


def test_final_cell_regression_veto_is_not_masked_by_aggregate_improvement():
    from zocr.consensus.cell_evaluation import compare_cell_evaluations

    old = {"predictions": [{"id": "a", "expected": "2358", "predicted": "2358"},
                           {"id": "b", "expected": "00000000", "predicted": "xxxxxxxx"}]}
    new = {"predictions": [{"id": "a", "expected": "2358", "predicted": "5555"},
                           {"id": "b", "expected": "00000000", "predicted": "00000000"}]}
    decision = compare_cell_evaluations(old, new)
    assert decision["recommended_action"] == "reject"
    assert decision["regressions"] == ["a"]
    assert not decision["activation_supported"]
    with pytest.raises(ValueError, match="identities"):
        compare_cell_evaluations(old, {"predictions": []})


def test_same_document_with_different_group_name_is_not_independent(tmp_path):
    path, _ = cell_dataset(tmp_path)
    document_hash = hashlib.sha256((tmp_path / "cell.png").read_bytes()).hexdigest()
    memory = {"learning_trial": {"document_lineage_complete": True, "training_document_sha256": [document_hash]}}
    with pytest.raises(ValueError, match="used for training"):
        load_cell_dataset(path, memory)


def test_document_family_and_incomplete_lineage_are_rejected(tmp_path):
    path, row = cell_dataset(tmp_path)
    with pytest.raises(ValueError, match="lineage"):
        load_cell_dataset(path, {})
    memory = {"learning_trial": {"document_lineage_complete": True, "training_document_sha256": ["0" * 64],
                                 "selection_group_ids": [row["group_id"]]}}
    with pytest.raises(ValueError, match="used for training"):
        load_cell_dataset(path, memory)


def test_full_cell_evaluation_uses_real_ocr_in_fresh_process(tmp_path):
    path, _ = cell_dataset(tmp_path)
    report = evaluate_cells(path, tmp_path / "evaluation", allowed_chars="2358", timeout_sec=60)
    assert report["scope"] == "full_cell_ocr"
    assert report["metrics"]["samples"] == 1
    assert report["real_document_samples"] == 0
    assert report["predictions"][0]["effective_model_sha256"]
    assert (tmp_path / "evaluation" / "cell_evaluation.json").exists()
    with pytest.raises(FileExistsError):
        evaluate_cells(path, tmp_path / "evaluation")


def test_frozen_memory_parameters_survive_loading_exactly(tmp_path):
    from zocr.consensus import toy_runtime

    path, _ = cell_dataset(tmp_path)
    memory = toy_runtime._toy_memory_payload()
    memory["learning_trial"] = {"document_lineage_complete": True, "training_document_sha256": ["a" * 64]}
    report = evaluate_cells(path, tmp_path / "frozen", memory=memory, allowed_chars="2358", timeout_sec=60)
    assert report["model"] == "explicit_memory"
    assert report["metrics"]["reference_characters"] == 4
