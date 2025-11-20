from __future__ import annotations

import json
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
        model = _load_sentence_transformer(model_name_or_path)

        def encode_fn(batch: Iterable[str]):  # type: ignore[redefined-outer-name]
            return model.encode(
                list(batch),
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
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
