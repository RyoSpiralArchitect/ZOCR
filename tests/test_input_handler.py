# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024 ZOCR contributors

import pytest

from zocr.ocr_pipeline import BasicInputHandler, DocumentInput


def test_basic_input_handler_with_images():
    images = [object(), object()]
    handler = BasicInputHandler()

    pages = handler.load(DocumentInput(document_id="doc-1", images=images, dpi=300))

    assert [p.page_number for p in pages] == [1, 2]
    assert [p.image for p in pages] == images
    assert all(p.document_id == "doc-1" for p in pages)
    assert all(p.dpi == 300 for p in pages)


def test_basic_input_handler_requires_pdf_dependency(tmp_path, monkeypatch):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%%EOF")

    handler = BasicInputHandler()

    with pytest.raises(RuntimeError) as err:
        handler.load(DocumentInput(document_id="doc-2", file_path=str(pdf_path)))

    assert "pdf2image" in str(err.value)

