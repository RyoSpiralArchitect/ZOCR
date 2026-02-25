from __future__ import annotations

import json

from PIL import Image


def test_toy_runtime_imports() -> None:
    from zocr.consensus import toy_runtime

    assert toy_runtime._cc_label_rle is not None
    assert isinstance(toy_runtime._GLYPH_ATLAS, dict)
    assert isinstance(toy_runtime._GLYPH_FEATS, dict)


def test_glyph_atlas_contains_digits() -> None:
    from zocr.consensus import toy_runtime

    atlas = toy_runtime._GLYPH_ATLAS
    for ch in "0123456789":
        assert ch in atlas
        assert atlas[ch]


def test_match_glyph_self_template() -> None:
    from zocr.consensus import toy_runtime

    atlas = toy_runtime._GLYPH_ATLAS
    glyph = atlas["5"][0]
    ch, conf = toy_runtime._match_glyph(glyph, atlas, allowed_chars="0123456789")
    assert ch == "5"
    assert conf >= 0.52


def test_match_glyph_respects_allowed_chars() -> None:
    from zocr.consensus import toy_runtime

    atlas = toy_runtime._GLYPH_ATLAS
    glyph = atlas["A"][0]
    unrestricted_ch, unrestricted_conf = toy_runtime._match_glyph(glyph, atlas)
    restricted_ch, restricted_conf = toy_runtime._match_glyph(glyph, atlas, allowed_chars="0123456789")
    assert unrestricted_ch == "A"
    assert unrestricted_conf >= 0.6
    assert restricted_ch in "0123456789"
    assert restricted_conf >= 0.52


def test_template_library_contains_ascii_presets() -> None:
    from zocr.consensus import toy_runtime

    for token in ("item", "qty", "unit price", "amount", "total"):
        assert token in toy_runtime._TOKEN_TEMPLATE_LIBRARY
        assert toy_runtime._TOKEN_TEMPLATE_LIBRARY[token]


def test_contextual_variants_do_not_hallucinate_from_question_mark() -> None:
    from zocr.consensus import toy_runtime

    assert toy_runtime._generate_contextual_variants("?") == set()


def test_export_jsonl_with_ocr_smoke(tmp_path) -> None:
    from zocr.consensus import toy_runtime

    img_path = tmp_path / "page.png"
    Image.new("RGB", (12, 12), (255, 255, 255)).save(img_path)

    doc_path = tmp_path / "doc.zocr.json"
    doc_path.write_text(json.dumps({"doc_id": "doc", "pages": []}), encoding="utf-8")

    out_path = tmp_path / "out.jsonl"
    n = toy_runtime.export_jsonl_with_ocr(
        str(doc_path),
        str(img_path),
        str(out_path),
        ocr_engine="toy",
        contextual=True,
    )
    assert n == 0
    assert out_path.exists()
