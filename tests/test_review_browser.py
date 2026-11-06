import json
import os

import pytest
from PIL import Image, ImageDraw

from zocr.diff.differ import SemanticDiffer
from zocr.diff.review import render_review


def test_review_decision_download_in_browser(tmp_path):
    if os.environ.get("ZOCR_TEST_BROWSER") != "1":
        pytest.skip("Set ZOCR_TEST_BROWSER=1 after installing Playwright Chromium")
    playwright = pytest.importorskip("playwright.sync_api")
    paths = []
    for side, amount in (("a", "100"), ("b", "120")):
        image = Image.new("RGB", (160, 60), "white")
        ImageDraw.Draw(image).text((10, 10), "Amount " + amount, fill="black")
        image.save(tmp_path / f"{side}.png")
        path = tmp_path / f"{side}.jsonl"
        path.write_text(json.dumps({
            "doc_id": "invoice", "page": 1, "table_index": 0, "row": 1, "col": 0,
            "text": amount, "header": "Amount", "bbox": [0, 0, 160, 60],
            "image_path": f"{side}.png", "meta": {"confidence": .9},
        }) + "\n", encoding="utf-8")
        paths.append(path)
    result = SemanticDiffer().compare_bundle(*paths)
    output = tmp_path / "review.html"
    render_review(result, output, cells_a=paths[0], cells_b=paths[1])
    errors, dialogs = [], []
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch()
        page = browser.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))

        def accept(dialog):
            dialogs.append(dialog.message)
            dialog.accept()

        page.on("dialog", accept)
        page.goto(output.as_uri())
        assert page.locator("article").count() == 1
        assert page.locator("img").count() == 2
        page.get_by_text("Export review decisions", exact=True).click()
        assert dialogs[-1] == "Enter a reviewer name."
        page.locator("#reviewer").fill("Test Reviewer")
        page.locator("select").select_option("approved")
        page.get_by_text("Export review decisions", exact=True).click()
        assert dialogs[-1] == "Add a reason for each reviewed decision."
        page.locator("textarea").fill("Verified both original amount fields.")
        with page.expect_download() as download:
            page.get_by_text("Export review decisions", exact=True).click()
        downloaded = json.loads(download.value.path().read_text(encoding="utf-8"))
        assert downloaded["schema"] == "zocr.diff_decisions.v1"
        assert downloaded["reviewer"] == "Test Reviewer"
        assert downloaded["source_hashes"] == result["review"]["source_hashes"]
        assert downloaded["decisions"][0]["event_id"] == result["events"][0]["event_id"]
        assert downloaded["decisions"][0]["status"] == "approved"
        assert not errors
        page.screenshot(path=str(tmp_path / "review.png"), full_page=True)
        browser.close()
