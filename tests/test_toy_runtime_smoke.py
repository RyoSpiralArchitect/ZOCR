from __future__ import annotations

import json

from PIL import Image, ImageDraw


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


def test_text_from_binary_cache_respects_allowed_chars() -> None:
    from zocr.consensus import toy_runtime

    toy_runtime._GLYPH_RUNTIME_CACHE.clear()
    glyph = toy_runtime._GLYPH_ATLAS["A"][0]
    arr = toy_runtime.np.asarray(glyph, dtype=toy_runtime.np.uint8)
    unrestricted_txt, _ = toy_runtime._text_from_binary(arr)
    restricted_txt, _ = toy_runtime._text_from_binary(arr, allowed_chars="0123456789")
    assert unrestricted_txt == "A"
    assert restricted_txt in "0123456789"


def test_text_from_binary_merges_colon_fragments() -> None:
    from zocr.consensus import toy_runtime

    glyph = toy_runtime._GLYPH_ATLAS[":"][0]
    arr = toy_runtime.np.asarray(glyph, dtype=toy_runtime.np.uint8)
    bw = (arr > 32).astype(toy_runtime.np.uint8) * 255
    txt, _ = toy_runtime._text_from_binary(bw, allowed_chars=":")
    assert txt == ":"


def test_restore_digit_commas_by_headers() -> None:
    from zocr.consensus import toy_runtime

    grid_text = [
        ["1,1", "2,2"],
        ["11", "22"],
        ["1.1", "2.2"],
    ]
    grid_conf = [
        [0.9, 0.9],
        [0.9, 0.9],
        [0.9, 0.9],
    ]
    notes = {}
    changed = toy_runtime._restore_digit_commas_by_headers(
        grid_text[0],
        grid_text,
        grid_conf=grid_conf,
        col_charset_hints=["0123456789,", "0123456789,"],
        fallback_notes=notes,
    )
    assert changed == 4
    assert grid_text[1] == ["1,1", "2,2"]
    assert grid_text[2] == ["1,1", "2,2"]
    assert notes[(1, 0)] == "comma_restore"
    assert grid_conf[1][0] < 0.9


def test_numeric_header_kinds_skips_coordinate_headers() -> None:
    from zocr.consensus import toy_runtime

    grid_text = [
        ["1,1"],
        ["11"],
        ["22"],
    ]
    kinds = toy_runtime._numeric_header_kinds(grid_text[0], grid_text)
    assert kinds and kinds[0] is None


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


def test_infer_row_bands_from_projection_detects_rows() -> None:
    from zocr.consensus import toy_runtime

    img = Image.new("RGB", (220, 220), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for idx in range(5):
        y = 18 + idx * 36
        draw.rectangle((25, y, 195, y + 7), fill=(0, 0, 0))
    bands = toy_runtime._infer_row_bands_from_table_projection(img, y_offset=0)
    assert len(bands) == 5


def test_export_uses_row_projection_when_undersegmented(tmp_path) -> None:
    from zocr.consensus import toy_runtime

    x1, y1, x2, y2 = 10, 10, 210, 210
    page = Image.new("RGB", (220, 220), (255, 255, 255))
    draw = ImageDraw.Draw(page)
    for idx in range(5):
        y = y1 + 15 + idx * 35
        draw.rectangle((x1 + 10, y, x2 - 10, y + 7), fill=(0, 0, 0))
    img_path = tmp_path / "page.png"
    page.save(img_path)

    doc_path = tmp_path / "doc.zocr.json"
    doc_path.write_text(
        json.dumps(
            {
                "doc_id": "doc",
                "pages": [
                    {
                        "index": 0,
                        "tables": [
                            {
                                "bbox": [x1, y1, x2, y2],
                                "dbg": {
                                    "col_bounds": [0, 100, 200],
                                    "baselines_segs": [[], []],
                                    "rows": 2,
                                },
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    out_path = tmp_path / "out.jsonl"
    n = toy_runtime.export_jsonl_with_ocr(
        str(doc_path),
        str(img_path),
        str(out_path),
        ocr_engine="toy",
        contextual=False,
    )
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert n == 10
    assert len(lines) == 10
