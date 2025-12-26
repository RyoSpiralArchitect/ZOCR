import pytest

from zocr.ocr_pipeline import BasicInputHandler, DocumentInput


def _write_simple_pdf(path):
    import fitz

    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 72), "hello world")
        doc.save(path.as_posix())


def test_basic_input_handler_with_images():
    images = [object(), object()]
    handler = BasicInputHandler()

    pages = handler.load(DocumentInput(document_id="doc-1", images=images, dpi=300))

    assert [p.page_number for p in pages] == [1, 2]
    assert [p.image for p in pages] == images
    assert all(p.document_id == "doc-1" for p in pages)
    assert all(p.dpi == 300 for p in pages)


def test_basic_input_handler_uses_pymupdf_fallback(tmp_path, monkeypatch):
    pdf_path = tmp_path / "sample.pdf"
    _write_simple_pdf(pdf_path)

    handler = BasicInputHandler()

    def _fail_pdf2image(self, path, dpi):
        raise RuntimeError("pdf2image missing")

    monkeypatch.setattr(BasicInputHandler, "_render_pdf_with_pdf2image", _fail_pdf2image)

    pages = handler.load(DocumentInput(document_id="doc-2", file_path=str(pdf_path), dpi=180))

    assert [p.page_number for p in pages] == [1]
    assert pages[0].document_id == "doc-2"
    assert getattr(pages[0].image, "size", None) is not None
    assert pages[0].dpi == 180


def test_basic_input_handler_surfaces_pdf_errors(tmp_path, monkeypatch):
    pdf_path = tmp_path / "sample.pdf"
    _write_simple_pdf(pdf_path)

    handler = BasicInputHandler()

    def _fail_pdf2image(self, path, dpi):
        raise RuntimeError("pdf2image missing")

    def _fail_pymupdf(self, path, dpi):
        raise RuntimeError("pymupdf missing")

    monkeypatch.setattr(BasicInputHandler, "_render_pdf_with_pdf2image", _fail_pdf2image)
    monkeypatch.setattr(BasicInputHandler, "_render_pdf_with_pymupdf", _fail_pymupdf)

    with pytest.raises(RuntimeError) as err:
        handler.load(DocumentInput(document_id="doc-3", file_path=str(pdf_path)))

    message = str(err.value)
    assert "pdf2image" in message
    assert "pymupdf" in message
