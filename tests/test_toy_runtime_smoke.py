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


def test_select_glyph_candidates_expands_when_margin_is_tight() -> None:
    from zocr.consensus import toy_runtime

    scored = [
        toy_runtime._GlyphCandidateScore("A", 0.810, 0.845, 0.31),
        toy_runtime._GlyphCandidateScore("B", 0.804, 0.838, 0.32),
        toy_runtime._GlyphCandidateScore("C", 0.799, 0.831, 0.33),
        toy_runtime._GlyphCandidateScore("D", 0.771, 0.805, 0.37),
    ]

    selected = toy_runtime._select_glyph_candidates(scored, threshold=0.6, top_limit=1)

    assert [glyph for glyph, _ in selected[:3]] == ["A", "B", "C"]
    assert len(selected) >= 3


def test_select_glyph_candidates_keeps_ambiguous_group_members() -> None:
    from zocr.consensus import toy_runtime

    scored = [
        toy_runtime._GlyphCandidateScore("0", 0.83, 0.86, 0.22),
        toy_runtime._GlyphCandidateScore("X", 0.72, 0.75, 0.91),
        toy_runtime._GlyphCandidateScore("O", 0.66, 0.69, 0.29),
        toy_runtime._GlyphCandidateScore("D", 0.64, 0.67, 0.31),
    ]

    selected = toy_runtime._select_glyph_candidates(scored, threshold=0.6, top_limit=1)
    selected_chars = [glyph for glyph, _ in selected]

    assert "0" in selected_chars
    assert "O" in selected_chars
    assert "D" in selected_chars


def test_select_glyph_candidates_rescues_feature_similar_shapes() -> None:
    from zocr.consensus import toy_runtime

    scored = [
        toy_runtime._GlyphCandidateScore("A", 0.84, 0.87, 0.20),
        toy_runtime._GlyphCandidateScore("C", 0.74, 0.78, 0.86),
        toy_runtime._GlyphCandidateScore("B", 0.66, 0.71, 0.24),
    ]

    selected = toy_runtime._select_glyph_candidates(scored, threshold=0.6, top_limit=1)
    selected_chars = [glyph for glyph, _ in selected]

    assert selected_chars == ["A", "B"]


def test_compute_glyph_features_tracks_centroid_and_balance() -> None:
    from zocr.consensus import toy_runtime

    arr = toy_runtime.np.zeros((6, 6), dtype=toy_runtime.np.float32)
    arr[:2, :2] = 1.0

    feats = toy_runtime._compute_glyph_features_from_array(arr)

    assert feats["center_x"] < 0.35
    assert feats["center_y"] < 0.35
    assert feats["h_balance"] < 0.0
    assert feats["v_balance"] < 0.0
    assert feats["row_peak"] > feats["density"]
    assert feats["col_peak"] > feats["density"]


def test_glyph_feature_distance_prefers_similar_layouts() -> None:
    from zocr.consensus import toy_runtime

    left_top = toy_runtime.np.zeros((6, 6), dtype=toy_runtime.np.float32)
    left_top[:2, :2] = 1.0
    left_top_wide = toy_runtime.np.zeros((6, 6), dtype=toy_runtime.np.float32)
    left_top_wide[:2, :3] = 1.0
    right_bottom = toy_runtime.np.zeros((6, 6), dtype=toy_runtime.np.float32)
    right_bottom[4:, 4:] = 1.0

    base_feats = toy_runtime._compute_glyph_features_from_array(left_top)
    near_feats = toy_runtime._compute_glyph_features_from_array(left_top_wide)
    far_feats = toy_runtime._compute_glyph_features_from_array(right_bottom)

    near = toy_runtime._glyph_feature_distance(base_feats, near_feats)
    far = toy_runtime._glyph_feature_distance(base_feats, far_feats)

    assert near["centroid_delta"] < far["centroid_delta"]
    assert near["balance_delta"] < far["balance_delta"]
    assert near["distance"] < far["distance"]


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


def test_self_augment_views_closing_merges_components() -> None:
    from zocr.consensus import toy_runtime

    bw = toy_runtime.np.zeros((10, 10), dtype=toy_runtime.np.uint8)
    bw[2:5, 1:3] = 255
    bw[2:5, 4:6] = 255
    assert len(toy_runtime._cc_label_rle(bw)) == 2

    close_bw = None
    for aug, meta in toy_runtime._self_augment_views(None, bw):
        if meta.get("type") == "augment_close" and meta.get("size") == 3:
            close_bw = aug
            break
    assert close_bw is not None
    assert len(toy_runtime._cc_label_rle(close_bw)) == 1


def test_self_augment_views_opening_removes_speckle() -> None:
    from zocr.consensus import toy_runtime

    bw = toy_runtime.np.zeros((10, 10), dtype=toy_runtime.np.uint8)
    bw[2:6, 2:6] = 255
    bw[1, 1] = 255
    assert len(toy_runtime._cc_label_rle(bw)) == 2

    open_bw = None
    for aug, meta in toy_runtime._self_augment_views(None, bw):
        if meta.get("type") == "augment_open" and meta.get("size") == 3:
            open_bw = aug
            break
    assert open_bw is not None
    assert len(toy_runtime._cc_label_rle(open_bw)) == 1


def test_refine_component_segments_never_drops_component() -> None:
    from zocr.consensus import toy_runtime

    bw = toy_runtime.np.zeros((4, 8), dtype=toy_runtime.np.uint8)
    bw[1:3, 1:7] = 255
    bbox = (1, 1, 7, 3, float((7 - 1) * (3 - 1)))
    refined = toy_runtime._refine_component_segments(bw, bbox)
    assert refined


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


def test_segmentation_candidates_keep_split_and_unsplit_variants() -> None:
    from zocr.consensus import toy_runtime

    bw = toy_runtime.np.zeros((12, 14), dtype=toy_runtime.np.uint8)
    bw[2:10, 1:4] = 255
    bw[2:10, 8:11] = 255
    bw[6, 4:8] = 255
    bbox = (1, 2, 11, 10, float(toy_runtime.np.count_nonzero(bw[2:10, 1:11])))
    baseline = toy_runtime._BaselineStats(
        baseline=10.0,
        xheight=8.0,
        ascender=8.0,
        descender=1.0,
        avg_width=3.0,
        avg_height=8.0,
        stroke_density=0.42,
        aspect_median=0.4,
    )

    candidates = toy_runtime._build_segmentation_sequence_candidates(bw, [bbox], baseline, beam_limit=4)
    lengths = {len(seq) for seq in candidates}

    assert 1 in lengths
    assert 2 in lengths


def test_decode_component_sequence_uses_glyph_beam_rerank() -> None:
    from zocr.consensus import toy_runtime

    bw = toy_runtime.np.zeros((2, 4), dtype=toy_runtime.np.uint8)
    bw[:, :2] = toy_runtime.np.asarray([[255, 0], [0, 255]], dtype=toy_runtime.np.uint8)
    bw[:, 2:] = toy_runtime.np.asarray([[0, 255], [255, 0]], dtype=toy_runtime.np.uint8)
    boxes = [(0, 0, 2, 2, 2.0), (2, 0, 4, 2, 2.0)]

    old_match = toy_runtime._match_glyph_candidates
    old_quality = toy_runtime._toy_text_quality
    old_topk = toy_runtime._TOY_GLYPH_CANDIDATE_TOPK
    old_beam = toy_runtime._TOY_GLYPH_BEAM
    calls = {"count": 0}

    def fake_match(_patch, _atlas, allowed_chars=None, top_k=None):
        calls["count"] += 1
        if calls["count"] == 1:
            return [("X", 0.93), ("A", 0.91)]
        return [("Y", 0.94), ("B", 0.92)]

    def fake_quality(text: str):
        if text == "AB":
            return 1.3, {"reason": "beam"}
        return 0.6, {"reason": "base"}

    try:
        toy_runtime._match_glyph_candidates = fake_match
        toy_runtime._toy_text_quality = fake_quality
        toy_runtime._TOY_GLYPH_CANDIDATE_TOPK = 2
        toy_runtime._TOY_GLYPH_BEAM = 2

        candidate = toy_runtime._decode_component_sequence(bw, boxes, allowed_chars="ABXY")

        assert candidate.text == "AB"
        assert [glyph.ch for glyph in candidate.glyphs] == ["A", "B"]
    finally:
        toy_runtime._match_glyph_candidates = old_match
        toy_runtime._toy_text_quality = old_quality
        toy_runtime._TOY_GLYPH_CANDIDATE_TOPK = old_topk
        toy_runtime._TOY_GLYPH_BEAM = old_beam


def test_observe_token_template_requires_repeated_support() -> None:
    from zocr.consensus import toy_runtime

    token = "ZocrDiscoveryAlpha42"
    bmp = toy_runtime._render_template_bitmap(token)
    assert bmp is not None
    arr = toy_runtime.np.asarray(bmp, dtype=toy_runtime.np.uint8)

    old_min_support = toy_runtime._TOKEN_TEMPLATE_REVIEW_MIN_SUPPORT
    old_min_conf = toy_runtime._TOKEN_TEMPLATE_REVIEW_MIN_CONF
    old_min_quality = toy_runtime._TOKEN_TEMPLATE_REVIEW_MIN_QUALITY
    old_review_state = toy_runtime.OrderedDict()
    for key, value in toy_runtime._TOKEN_TEMPLATE_REVIEW_STATE.items():
        copied = dict(value)
        variants = value.get("variants")
        if isinstance(variants, toy_runtime.deque):
            copied["variants"] = toy_runtime.deque(list(variants), maxlen=variants.maxlen)
        old_review_state[key] = copied
    old_stats = dict(toy_runtime._GLYPH_RUNTIME_STATS)
    old_template_state = dict(toy_runtime._TEMPLATE_CACHE_STATE)
    had_token = token in toy_runtime._TOKEN_TEMPLATE_LIBRARY
    old_token_variants = toy_runtime.deque(
        list(toy_runtime._TOKEN_TEMPLATE_LIBRARY.get(token, [])),
        maxlen=toy_runtime._TOKEN_TEMPLATE_MAX_VARIANTS,
    )
    try:
        toy_runtime._TOKEN_TEMPLATE_REVIEW_MIN_SUPPORT = 2
        toy_runtime._TOKEN_TEMPLATE_REVIEW_MIN_CONF = 0.0
        toy_runtime._TOKEN_TEMPLATE_REVIEW_MIN_QUALITY = 0.0
        toy_runtime._TOKEN_TEMPLATE_REVIEW_STATE.clear()
        toy_runtime._GLYPH_RUNTIME_STATS.clear()
        toy_runtime._TOKEN_TEMPLATE_LIBRARY.pop(token, None)
        toy_runtime._TEMPLATE_CACHE_STATE["dirty"] = False

        toy_runtime._observe_token_template(token, arr, confidence=0.95, quality=1.0)
        assert token not in toy_runtime._TOKEN_TEMPLATE_LIBRARY
        assert len(toy_runtime._TOKEN_TEMPLATE_REVIEW_STATE) == 1

        toy_runtime._observe_token_template(token, arr, confidence=0.95, quality=1.0)
        assert token in toy_runtime._TOKEN_TEMPLATE_LIBRARY
        assert len(toy_runtime._TOKEN_TEMPLATE_REVIEW_STATE) == 0
        match_token, match_conf = toy_runtime._match_token_template_from_cache(arr)
        assert match_token == token
        assert match_conf >= 0.45
        assert toy_runtime._GLYPH_RUNTIME_STATS["template_review_accepted"] >= 1.0
        assert toy_runtime._GLYPH_RUNTIME_STATS["template_discovered"] >= 1.0
    finally:
        toy_runtime._TOKEN_TEMPLATE_REVIEW_MIN_SUPPORT = old_min_support
        toy_runtime._TOKEN_TEMPLATE_REVIEW_MIN_CONF = old_min_conf
        toy_runtime._TOKEN_TEMPLATE_REVIEW_MIN_QUALITY = old_min_quality
        toy_runtime._TOKEN_TEMPLATE_REVIEW_STATE.clear()
        toy_runtime._TOKEN_TEMPLATE_REVIEW_STATE.update(old_review_state)
        toy_runtime._GLYPH_RUNTIME_STATS.clear()
        toy_runtime._GLYPH_RUNTIME_STATS.update(old_stats)
        toy_runtime._TEMPLATE_CACHE_STATE.clear()
        toy_runtime._TEMPLATE_CACHE_STATE.update(old_template_state)
        if had_token:
            toy_runtime._TOKEN_TEMPLATE_LIBRARY[token] = old_token_variants
        else:
            toy_runtime._TOKEN_TEMPLATE_LIBRARY.pop(token, None)


def test_generic_postprocess_policy_disables_table_adapters() -> None:
    from zocr.consensus import toy_runtime

    policy = toy_runtime._resolve_toy_postprocess_policy("generic", contextual=True)
    assert policy.name == "generic"
    assert not policy.use_header_charset_hints
    assert not policy.use_numeric_headers
    assert not policy.use_date_headers
    assert not policy.use_schema_rectifier
    assert not policy.use_footer_reflow
    assert not policy.use_comma_restore


def test_adapt_glyph_requires_review_support() -> None:
    from zocr.consensus import toy_runtime

    sample = toy_runtime._GLYPH_ATLAS["Z"][0]
    arr = toy_runtime.np.zeros((sample.height + 1, sample.width + 1), dtype=toy_runtime.np.uint8)
    arr[1:, 1:] = toy_runtime.np.asarray(sample, dtype=toy_runtime.np.uint8)
    candidate = Image.fromarray(arr)

    old_min_support = toy_runtime._GLYPH_REVIEW_MIN_SUPPORT
    old_min_conf = toy_runtime._GLYPH_REVIEW_MIN_CONF
    old_review_state = toy_runtime.OrderedDict(
        (key, dict(value)) for key, value in toy_runtime._GLYPH_REVIEW_STATE.items()
    )
    old_stats = dict(toy_runtime._GLYPH_RUNTIME_STATS)
    had_test_char = "@" in toy_runtime._GLYPH_ATLAS
    old_test_char = list(toy_runtime._GLYPH_ATLAS.get("@", []))
    old_test_feats = dict(toy_runtime._GLYPH_FEATS.get("@", {}))
    try:
        toy_runtime._GLYPH_REVIEW_MIN_SUPPORT = 2
        toy_runtime._GLYPH_REVIEW_MIN_CONF = 0.0
        toy_runtime._GLYPH_REVIEW_STATE.clear()
        toy_runtime._GLYPH_RUNTIME_STATS.clear()
        toy_runtime._GLYPH_ATLAS.pop("@", None)
        toy_runtime._GLYPH_FEATS.pop("@", None)

        toy_runtime._adapt_glyph("@", candidate, conf=0.95)
        assert "@" not in toy_runtime._GLYPH_ATLAS
        assert len(toy_runtime._GLYPH_REVIEW_STATE) == 1

        toy_runtime._adapt_glyph("@", candidate, conf=0.95)
        assert "@" in toy_runtime._GLYPH_ATLAS
        assert len(toy_runtime._GLYPH_ATLAS["@"]) == 1
        assert toy_runtime._GLYPH_RUNTIME_STATS["review_accepted"] >= 1.0
    finally:
        toy_runtime._GLYPH_REVIEW_MIN_SUPPORT = old_min_support
        toy_runtime._GLYPH_REVIEW_MIN_CONF = old_min_conf
        toy_runtime._GLYPH_REVIEW_STATE.clear()
        toy_runtime._GLYPH_REVIEW_STATE.update(old_review_state)
        toy_runtime._GLYPH_RUNTIME_STATS.clear()
        toy_runtime._GLYPH_RUNTIME_STATS.update(old_stats)
        if had_test_char:
            toy_runtime._GLYPH_ATLAS["@"] = old_test_char
            toy_runtime._GLYPH_FEATS["@"] = old_test_feats
        else:
            toy_runtime._GLYPH_ATLAS.pop("@", None)
            toy_runtime._GLYPH_FEATS.pop("@", None)


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
