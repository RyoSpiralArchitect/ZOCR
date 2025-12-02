import pytest

from zocr.embeddings.multimodal import (
    EmbeddingBackends,
    FusionConfig,
    MultimodalEmbeddingService,
)


def test_concat_projection_fuses_embeddings_and_metadata():
    def text_encoder(batch):
        return [[1.0, float(i)] for i, _ in enumerate(batch)]

    def vision_encoder(batch):
        return [[10.0 + i, 20.0 + i] for i, _ in enumerate(batch)]

    fusion = FusionConfig(
        strategy="concat",
        projection=[
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
    )

    service = MultimodalEmbeddingService(
        backends=EmbeddingBackends(text_encoder=text_encoder, vision_encoder=vision_encoder),
        fusion=fusion,
    )

    objects = [
        {
            "id": "cell-1",
            "text": "hello",
            "image": b"img-1",
            "bbox": [0, 0, 50, 100],
            "page_width": 100,
            "page_height": 100,
            "reading_order": 0,
        },
        {
            "id": "cell-2",
            "text": "world",
            "image": b"img-2",
            "bbox": [25, 25, 100, 100],
            "page_width": 100,
            "page_height": 100,
            "reading_order": 1,
        },
    ]

    store = service.embed_objects(objects)

    assert len(store.entries) == 2
    assert store.entries[0]["meta"]["id"] == "cell-1"
    assert store.entries[1]["meta"]["id"] == "cell-2"

    fused_first = store.entries[0]["vector"]
    fused_second = store.entries[1]["vector"]

    # Projection selects text[0] and the reading order component respectively.
    assert fused_first == [1.0, 0.0]
    assert fused_second == [1.0, 1.0]

    # Layout embedding preserves normalized coordinates and reading order.
    layout_first = store.entries[0]["embeddings"]["layout"]
    assert pytest.approx(layout_first[:4]) == [0.0, 0.0, 0.5, 1.0]
    assert layout_first[4] == 0.0


def test_late_fusion_with_layout_fallback():
    def text_encoder(batch):
        return [[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]]

    def extra_layout(obj, idx, total):
        return [float(idx), float(total)]

    service = MultimodalEmbeddingService(
        backends=EmbeddingBackends(
            text_encoder=text_encoder,
            vision_encoder=None,
            layout_feature_fn=extra_layout,
        ),
        fusion=FusionConfig(strategy="late_fusion", weights={"text": 1.0, "layout": 0.5}),
    )

    objects = [
        {
            "id": "a",
            "text": "foo",
            "bbox": [0, 0, 10, 10],
            "page_size": {"width": 10, "height": 10},
        },
        {
            "id": "b",
            "text": "bar",
            "bbox": [5, 5, 10, 10],
            "page_size": {"width": 10, "height": 10},
            "reading_order": 1,
        },
    ]

    store = service.embed_objects(objects)

    assert len(store.entries) == 2
    first_vec = store.entries[0]["vector"]
    second_vec = store.entries[1]["vector"]

    # Late fusion pads text (len=3) and layout (len=7) to the same length.
    assert len(first_vec) == 7
    assert len(second_vec) == 7

    # With weights text=1.0, layout=0.5 the fused vector averages modalities.
    expected_first = [
        (0.5 + 0.0 * 0.5) / 1.5,
        (0.5 + 0.0 * 0.5) / 1.5,
        (0.5 + 1.0 * 0.5) / 1.5,
    ]
    assert pytest.approx(first_vec[:3]) == expected_first
    assert store.entries[0]["meta"]["id"] == "a"
    assert store.entries[1]["meta"]["id"] == "b"
