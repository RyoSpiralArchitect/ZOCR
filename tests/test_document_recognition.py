import json

from PIL import Image

from zocr.ingest.document_recognition import prepare_image_pages


class Recognizer:
    def __init__(self, text="デザイナー", confidence=.99, languages=None):
        self.text, self.confidence = text, confidence
        self.languages = languages or ["ja-JP", "en-US"]
        self.calls = 0

    def recognize(self, image):
        self.calls += 1
        return {"engine": "apple_vision", "revision": 3, "languages": self.languages,
                "lines": [{"text": self.text, "confidence": self.confidence,
                           "bbox": [1, 2, image.width - 1, image.height - 2]}]}


def test_image_ocr_preserves_evidence_and_raw_hypotheses_when_abstaining(tmp_path):
    image_path = tmp_path / "scan.png"
    Image.new("RGB", (200, 100), "white").save(image_path)
    recognizer = Recognizer("mxaxW+", .99, ["en-US"])
    plan = prepare_image_pages({0: str(image_path)}, {}, tmp_path / "out", recognizer=recognizer)[0]
    assert plan["route"] == "vision"
    record, = plan["records"]
    assert record["text"] == ""
    assert record["meta"]["observed_text"] == "mxaxW+"
    assert record["meta"]["recognition_status"] == "abstained"
    assert record["meta"]["abstention_reasons"] == ["unsupported_language"]
    assert record["meta"]["needs_review"]
    assert record["table_index"] is None
    assert recognizer.calls == 1


def test_pure_native_pages_do_not_initialize_image_recognition(tmp_path):
    native = {0: {"route": "native", "records": [{"text": "原文"}]}}
    recognizer = Recognizer()
    plans = prepare_image_pages({0: "not-opened.png"}, native, tmp_path / "out", recognizer=recognizer)
    assert plans == native
    assert recognizer.calls == 0


def test_mixed_page_keeps_native_text_and_offsets_image_ocr_boxes(tmp_path):
    image_path = tmp_path / "mixed.png"
    Image.new("RGB", (200, 100), "white").save(image_path)
    native_record = {"text": "原文", "meta": {"source_kind": "pdf_text_layer"}}
    native = {0: {"route": "mixed", "records": [native_record], "ocr_regions": [[50, 20, 150, 80]]}}
    plan = prepare_image_pages({0: str(image_path)}, native, tmp_path / "out", recognizer=Recognizer())[0]
    assert plan["route"] == "mixed_vision"
    assert plan["records"][0] == native_record
    record = plan["records"][1]
    assert record["bbox"] == [51, 22, 149, 78]
    assert record["text"] == "デザイナー"
    assert record["meta"]["source_kind"] == "image_ocr"
    json.dumps(plan, allow_nan=False)


def test_all_abstained_records_block_publication_even_with_high_raw_confidence(tmp_path):
    from zocr.artifacts.quality import quality_summary, write_final_quality

    image_path = tmp_path / "scan.png"
    Image.new("RGB", (200, 100), "white").save(image_path)
    out = tmp_path / "out"
    plan = prepare_image_pages({0: str(image_path)}, {}, out,
                               recognizer=Recognizer("mxaxW+", .99, ["en-US"]))[0]
    (out / "rag").mkdir()
    payload = "\n".join(json.dumps(record) for record in plan["records"]) + "\n"
    for name in ("doc.contextual.jsonl", "doc.mm.jsonl", "rag/cells.jsonl"):
        (out / name).write_text(payload)
    write_final_quality(out, min_confidence=.58)
    assert quality_summary(out)["quality_status"] == "fail"
    assert quality_summary(out)["publication_status"] == "blocked"
