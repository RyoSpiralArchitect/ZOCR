# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024 ZOCR contributors

import builtins
import json
import sys

import pytest

from zocr.core.embedders import embed_jsonl


def _stub_encode(batch):
    return [[float(len(text)), float(i)] for i, text in enumerate(batch)]


def test_embed_jsonl_writes_vectors(tmp_path):
    src = tmp_path / "cells.jsonl"
    rows = [
        {"text": "hello world", "page": 1},
        {"text": "another line", "page": 2},
    ]
    with src.open("w", encoding="utf-8") as fw:
        for row in rows:
            fw.write(json.dumps(row, ensure_ascii=False) + "\n")

    out_path = embed_jsonl(src, None, model_name_or_path="stub", encode_fn=_stub_encode)
    assert out_path.endswith(".embedded.jsonl")

    with open(out_path, "r", encoding="utf-8") as fr:
        enriched = [json.loads(line) for line in fr]

    assert len(enriched) == 2
    assert enriched[0]["embeddings"]["text"] == [11.0, 0.0]
    assert enriched[1]["embeddings"]["text"] == [12.0, 1.0]


def test_embed_jsonl_requires_dependency(monkeypatch, tmp_path):
    src = tmp_path / "cells.jsonl"
    src.write_text(json.dumps({"text": "hi"}), encoding="utf-8")

    def _raise_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("blocked")
        return original_import(name, *args, **kwargs)

    original_import = builtins.__import__
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    monkeypatch.setattr(builtins, "__import__", _raise_import)
    try:
        with pytest.raises(RuntimeError):
            embed_jsonl(src, tmp_path / "out.jsonl", model_name_or_path="stub")
    finally:
        monkeypatch.setattr(builtins, "__import__", original_import)


def test_embed_jsonl_bedrock(monkeypatch, tmp_path):
    class DummyBody:
        def __init__(self, payload):
            self.payload = json.dumps(payload).encode()

        def read(self):
            return self.payload

    class DummyClient:
        def __init__(self):
            self.calls = []

        def invoke_model(self, modelId, body, **kwargs):
            data = json.loads(body)
            self.calls.append((modelId, data))
            txt = data.get("inputText", "")
            return {"body": DummyBody({"embedding": [1.0, float(len(txt))]})}

    class DummySession:
        def __init__(self, *args, **kwargs):
            self.client_obj = DummyClient()

        def client(self, *_args, **_kwargs):
            return self.client_obj

    dummy_client = DummyClient()

    class DummyBoto:
        Session = DummySession

        @staticmethod
        def client(*_args, **_kwargs):
            return dummy_client

    monkeypatch.setitem(sys.modules, "boto3", DummyBoto)

    src = tmp_path / "cells.jsonl"
    rows = [
        {"text": "short"},
        {"text": "longer text"},
    ]
    with src.open("w", encoding="utf-8") as fw:
        for row in rows:
            fw.write(json.dumps(row) + "\n")

    out = embed_jsonl(
        src,
        None,
        model_name_or_path="bedrock-model",
        provider="bedrock",
        aws_region="us-east-1",
        normalize=False,
    )

    with open(out, "r", encoding="utf-8") as fr:
        enriched = [json.loads(line) for line in fr]

    assert enriched[0]["embeddings"]["text"] == [1.0, 5.0]
    assert enriched[1]["embeddings"]["text"] == [1.0, 11.0]
    assert len(dummy_client.calls) == 2


def test_embed_jsonl_bedrock_requires_boto(monkeypatch, tmp_path):
    src = tmp_path / "cells.jsonl"
    src.write_text(json.dumps({"text": "hi"}), encoding="utf-8")

    monkeypatch.setitem(sys.modules, "boto3", None)

    with pytest.raises(RuntimeError):
        embed_jsonl(
            src,
            None,
            model_name_or_path="bedrock-model",
            provider="bedrock",
            aws_region="us-east-1",
        )
