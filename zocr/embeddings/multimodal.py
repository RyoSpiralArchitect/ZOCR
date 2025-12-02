from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

NumericVector = Sequence[float]
Encoder = Callable[[Sequence[Any]], Sequence[NumericVector]]


@dataclass
class EmbeddingBackends:
    """Container for optional modality-specific encoders."""

    text_encoder: Encoder
    vision_encoder: Optional[Encoder] = None
    layout_feature_fn: Optional[Callable[[Dict[str, Any], int, int], NumericVector]] = None
    vision_required: bool = False


@dataclass
class FusionConfig:
    """Configuration for combining modality vectors."""

    strategy: str = "concat"
    projection: Optional[Sequence[Sequence[float]]] = None
    weights: Dict[str, float] = field(
        default_factory=lambda: {"text": 1.0, "vision": 0.6, "layout": 0.4}
    )


class FusionStrategy:
    """Helper implementing multiple fusion strategies for embeddings."""

    def __init__(self, config: FusionConfig):
        self.config = config

    def fuse(
        self,
        *,
        text: NumericVector,
        vision: Optional[NumericVector],
        layout: NumericVector,
    ) -> List[float]:
        name = (self.config.strategy or "concat").lower()
        if name == "concat":
            return self._concat_with_projection(text=text, vision=vision, layout=layout)
        if name == "late_fusion":
            return self._late_fusion(text=text, vision=vision, layout=layout)
        raise ValueError(f"Unknown fusion strategy: {self.config.strategy}")

    def _concat_with_projection(
        self,
        *,
        text: NumericVector,
        vision: Optional[NumericVector],
        layout: NumericVector,
    ) -> List[float]:
        pieces: List[float] = list(text)
        if vision:
            pieces.extend(vision)
        pieces.extend(layout)

        if not self.config.projection:
            return pieces

        projected: List[float] = []
        for row in self.config.projection:
            if len(row) != len(pieces):
                raise ValueError("Projection matrix width must match concatenated vector length")
            projected.append(sum(a * b for a, b in zip(row, pieces)))
        return projected

    def _late_fusion(
        self,
        *,
        text: NumericVector,
        vision: Optional[NumericVector],
        layout: NumericVector,
    ) -> List[float]:
        max_len = max(len(text), len(vision or []), len(layout))
        padded: List[List[float]] = []
        weights: List[float] = []

        def _pad(vec: Sequence[float]) -> List[float]:
            return list(vec) + [0.0] * (max_len - len(vec))

        padded.append(_pad(text))
        weights.append(self.config.weights.get("text", 1.0))

        if vision:
            padded.append(_pad(vision))
            weights.append(self.config.weights.get("vision", 0.0))

        padded.append(_pad(layout))
        weights.append(self.config.weights.get("layout", 0.0))

        total = sum(weights) or 1.0
        fused = [0.0 for _ in range(max_len)]
        for vec, w in zip(padded, weights):
            for i, val in enumerate(vec):
                fused[i] += w * val
        return [v / total for v in fused]


class VectorStore:
    """Very small in-memory store capturing vectors and metadata."""

    def __init__(self) -> None:
        self.entries: List[Dict[str, Any]] = []

    def add(
        self,
        *,
        vector: Sequence[float],
        embeddings: Dict[str, Optional[NumericVector]],
        meta: Dict[str, Any],
    ) -> None:
        self.entries.append(
            {
                "vector": list(vector),
                "embeddings": embeddings,
                "meta": meta,
            }
        )


class MultimodalEmbeddingService:
    """Compute text/vision/layout embeddings with optional fusion."""

    def __init__(
        self,
        *,
        backends: EmbeddingBackends,
        fusion: Optional[FusionConfig] = None,
    ) -> None:
        if not backends.text_encoder:
            raise ValueError("text_encoder is required")
        self.backends = backends
        self.fusion = FusionStrategy(fusion or FusionConfig())

    def embed_objects(self, objects: Sequence[Dict[str, Any]]) -> VectorStore:
        text_inputs = [obj.get("text", "") for obj in objects]
        text_vecs = list(self.backends.text_encoder(text_inputs))

        if len(text_vecs) != len(objects):
            raise ValueError("Text encoder returned mismatched number of embeddings")

        vision_vecs: List[Optional[NumericVector]] = []
        if self.backends.vision_encoder:
            vision_inputs: List[Any] = [obj.get("image") for obj in objects]
            try:
                vision_outputs = list(self.backends.vision_encoder(vision_inputs))
            except Exception:
                if self.backends.vision_required:
                    raise
                vision_outputs = [None for _ in objects]
            if vision_outputs and len(vision_outputs) != len(objects):
                raise ValueError("Vision encoder returned mismatched number of embeddings")
            for vec in vision_outputs:
                vision_vecs.append(vec)
        else:
            vision_vecs = [None for _ in objects]

        store = VectorStore()
        total = len(objects)
        for idx, (obj, text_vec, vision_vec) in enumerate(zip(objects, text_vecs, vision_vecs)):
            layout_vec = self._layout_features(obj=obj, idx=idx, total=total)
            fused = self.fusion.fuse(text=text_vec, vision=vision_vec, layout=layout_vec)
            embeddings_block = {
                "text": list(text_vec),
                "vision": list(vision_vec) if vision_vec else None,
                "layout": list(layout_vec),
                "fused": fused,
            }
            store.add(vector=fused, embeddings=embeddings_block, meta=dict(obj))
        return store

    def _layout_features(self, *, obj: Dict[str, Any], idx: int, total: int) -> List[float]:
        bbox = obj.get("bbox") or [0.0, 0.0, 0.0, 0.0]
        if len(bbox) != 4:
            raise ValueError("bbox must contain four numbers")
        width = float(obj.get("page_width") or obj.get("page_size", {}).get("width", 1.0) or 1.0)
        height = float(obj.get("page_height") or obj.get("page_size", {}).get("height", 1.0) or 1.0)

        x0, y0, x1, y1 = [float(v) for v in bbox]
        layout_vec = [
            x0 / width if width else 0.0,
            y0 / height if height else 0.0,
            x1 / width if width else 0.0,
            y1 / height if height else 0.0,
            float(obj.get("reading_order", idx)) / float(max(total - 1, 1)),
        ]

        if self.backends.layout_feature_fn:
            extra = self.backends.layout_feature_fn(obj, idx, total)
            layout_vec = list(layout_vec) + [float(x) for x in extra]
        return layout_vec
