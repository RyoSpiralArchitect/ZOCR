# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024 ZOCR contributors

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

__all__ = ["embed_jsonl"]


def _load_sentence_transformer(model_name_or_path: str):
    """Lazy import helper with a friendly error message."""

    try:  # pragma: no cover - exercised indirectly in error-path test
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover - dependency missing in CI
        raise RuntimeError(
            "sentence-transformers is required for embedding; install it or pass a custom encoder"
        ) from exc
    return SentenceTransformer(model_name_or_path)


def embed_jsonl(
    jsonl: str | Path,
    out_jsonl: Optional[str | Path],
    model_name_or_path: str,
    *,
    text_field: str = "text",
    batch_size: int = 32,
    normalize: bool = True,
    provider: str = "sentence-transformers",
    aws_profile: Optional[str] = None,
    aws_region: Optional[str] = None,
    aws_endpoint_url: Optional[str] = None,
    encode_fn: Optional[Callable[[Iterable[str]], Sequence[Sequence[float]]]] = None,
) -> str:
    """Encode text chunks for downstream RAG consumers.

    The helper loads a SentenceTransformer model (unless ``encode_fn`` is supplied),
    encodes ``text_field`` from every JSONL record, and writes a copy with an
    ``embeddings.text`` vector attached.
    """

    src = Path(jsonl)
    if out_jsonl:
        dst = Path(out_jsonl)
    else:
        dst = src.with_suffix(src.suffix + ".embedded.jsonl")

    records: List[dict] = []
    with src.open("r", encoding="utf-8") as fr:
        for line in fr:
            if not line.strip():
                continue
            records.append(json.loads(line))

    texts = [str((rec.get(text_field) or "")) for rec in records]

    if encode_fn is None:
        encode_fn = _resolve_encoder(
            provider,
            model_name_or_path,
            batch_size=batch_size,
            normalize=normalize,
            aws_profile=aws_profile,
            aws_region=aws_region,
            aws_endpoint_url=aws_endpoint_url,
        )

    vectors = list(encode_fn(texts))  # type: ignore[arg-type]
    if len(vectors) != len(records):
        raise ValueError("Encoder returned a mismatched number of embeddings")

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as fw:
        for rec, vec in zip(records, vectors):
            emb_block = (rec.get("embeddings") or {}).copy()
            emb_block["text"] = [float(x) for x in vec]
            rec["embeddings"] = emb_block
            fw.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return str(dst)


def _resolve_encoder(
    provider: str,
    model_name_or_path: str,
    *,
    batch_size: int,
    normalize: bool,
    aws_profile: Optional[str],
    aws_region: Optional[str],
    aws_endpoint_url: Optional[str],
):
    provider = provider.lower()
    if provider in {"sentence-transformers", "st", "transformers"}:
        model = _load_sentence_transformer(model_name_or_path)

        def encode_fn(batch: Iterable[str]):
            return model.encode(
                list(batch),
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
            )

        return encode_fn

    if provider in {"bedrock", "aws-bedrock", "aws"}:
        return _bedrock_encoder(
            model_name_or_path,
            profile=aws_profile,
            region=aws_region,
            endpoint_url=aws_endpoint_url,
            normalize=normalize,
        )

    raise ValueError(f"Unknown embedding provider: {provider}")


def _bedrock_encoder(
    model_id: str,
    *,
    profile: Optional[str],
    region: Optional[str],
    endpoint_url: Optional[str],
    normalize: bool,
):
    client = _load_bedrock_client(profile=profile, region=region, endpoint_url=endpoint_url)

    def encode(batch: Iterable[str]):
        vectors = []
        for text in batch:
            payload = json.dumps({"inputText": text})
            response = client.invoke_model(
                modelId=model_id,
                body=payload,
                accept="application/json",
                contentType="application/json",
            )
            body = response.get("body")
            if hasattr(body, "read"):
                raw = body.read()
            else:
                raw = body
            parsed = json.loads(raw)
            vec = parsed.get("embedding") or parsed.get("embeddings") or parsed.get("vector")
            if isinstance(vec, dict):
                vec = vec.get("text") or vec.get("default")
            if vec is None:
                raise RuntimeError("Bedrock response missing embedding")
            floats = [float(x) for x in vec]
            if normalize:
                floats = _l2_normalize(floats)
            vectors.append(floats)
        return vectors

    return encode


def _load_bedrock_client(*, profile: Optional[str], region: Optional[str], endpoint_url: Optional[str]):
    try:  # pragma: no cover - exercised indirectly
        import boto3
    except Exception as exc:  # pragma: no cover - dependency missing in CI
        raise RuntimeError("boto3 is required for AWS Bedrock embedding") from exc

    if profile:
        session = boto3.Session(profile_name=profile, region_name=region)
        return session.client("bedrock-runtime", endpoint_url=endpoint_url)
    return boto3.client("bedrock-runtime", region_name=region, endpoint_url=endpoint_url)


def _l2_normalize(vec: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return [0.0 for _ in vec]
    return [float(x / norm) for x in vec]
