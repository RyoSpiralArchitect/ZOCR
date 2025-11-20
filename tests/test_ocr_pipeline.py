from zocr.ocr_pipeline import (
    DocumentInput,
    DocumentPipeline,
    MockAggregator,
    MockInputHandler,
    MockRegionClassifier,
    MockSegmenter,
    MockTableExtractor,
    MockTextOCR,
    MockVLLM,
    OcrPipeline,
    PageInput,
)


def test_pipeline_runs_with_mocks():
    segmenter = MockSegmenter()
    classifier = MockRegionClassifier()
    text_ocr = MockTextOCR()
    vllm = MockVLLM()
    table_extractor = MockTableExtractor()
    aggregator = MockAggregator()
    pipeline = OcrPipeline(
        segmenter=segmenter,
        region_classifier=classifier,
        text_ocr=text_ocr,
        vllm=vllm,
        table_extractor=table_extractor,
        aggregator=aggregator,
    )

    page = PageInput(document_id="doc-1", page_number=1, image="dummy")
    result = pipeline.process(page)

    assert segmenter.calls[0] == page
    assert {region.region_id for region in aggregator.calls[0]["classified_regions"]} == {
        "text-1",
        "image-1",
        "table-1",
    }
    assert result.metadata.total_regions == 3
    assert result.metadata.text_regions == 1
    assert result.metadata.image_regions == 1
    assert result.metadata.table_regions == 1
    assert [region.region_id for region in result.regions] == ["text-1", "image-1", "table-1"]
    assert result.regions[0].content == {"text": "text content for text-1", "confidence": 0.95}
    assert result.regions[1].content == {"caption": "caption for image-1", "confidence": 0.9}
    assert result.regions[2].content["table_data"]["headers"] == ["col1", "col2"]


def test_document_pipeline_processes_pages_in_order():
    input_handler = MockInputHandler()
    segmenter = MockSegmenter()
    classifier = MockRegionClassifier()
    text_ocr = MockTextOCR()
    vllm = MockVLLM()
    table_extractor = MockTableExtractor()
    aggregator = MockAggregator()
    page_pipeline = OcrPipeline(
        segmenter=segmenter,
        region_classifier=classifier,
        text_ocr=text_ocr,
        vllm=vllm,
        table_extractor=table_extractor,
        aggregator=aggregator,
    )

    pipeline = DocumentPipeline(input_handler=input_handler, page_pipeline=page_pipeline)

    images = [object(), object()]
    outputs = pipeline.process(DocumentInput(document_id="doc-1", images=images, dpi=200))

    assert len(outputs) == 2
    assert [call.document_id for call in input_handler.calls] == ["doc-1"]
    assert [result.page_number for result in outputs] == [1, 2]
    # Aggregator receives the correct page payload per invocation
    assert [payload["page"].page_number for payload in aggregator.calls] == [1, 2]


