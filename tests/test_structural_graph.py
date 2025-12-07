import json

import pytest

from zocr.ocr_pipeline import (
    BoundingBox,
    RegionOutput,
    RegionType,
    build_structural_graph,
    graph_from_jsonld,
    graph_to_jsonld,
    graph_to_llm_prompts,
    graph_to_tool_calls,
    load_jsonld,
    save_jsonld,
)


def _sample_regions():
    return [
        RegionOutput(
            region_id="text-1",
            type=RegionType.TEXT,
            bounding_box=BoundingBox(x=0, y=0, width=50, height=10),
            reading_order=0,
            content={"text": "Sample figure caption text"},
        ),
        RegionOutput(
            region_id="image-1",
            type=RegionType.IMAGE,
            bounding_box=BoundingBox(x=10, y=10, width=100, height=80),
            reading_order=1,
            content={"caption": "Sample figure caption text"},
        ),
        RegionOutput(
            region_id="table-1",
            type=RegionType.TABLE,
            bounding_box=BoundingBox(x=20, y=20, width=120, height=90),
            reading_order=2,
            content={
                "table_data": {"headers": ["Speed (rpm)", "Torque (Nm)"], "rows": []},
                "confidence": 0.9,
            },
        ),
    ]


def test_structural_graph_links_captions_and_headers(tmp_path):
    graph = build_structural_graph("doc-123", 1, _sample_regions())

    relations = {(edge.source, edge.target, edge.relation) for edge in graph.edges}
    assert ("image-1", "image-1#caption", "has_caption") in relations
    assert ("image-1#caption", "image-1", "describes") in relations
    assert any(edge.relation == "near_text" and edge.target == "text-1" for edge in graph.edges)
    assert any(edge.relation == "has_header" and edge.source == "table-1" for edge in graph.edges)

    table_node = next(node for node in graph.nodes if node.node_id == "table-1")
    assert set(table_node.attributes.get("units")) >= {"rpm", "Nm"}

    path = tmp_path / "structure.jsonld"
    save_jsonld(graph, path)
    restored = load_jsonld(path)

    restored_relations = {(edge.source, edge.target, edge.relation) for edge in restored.edges}
    assert relations == restored_relations

    json_payload = graph_to_jsonld(graph)
    round_trip = graph_from_jsonld(json.loads(json.dumps(json_payload)))
    assert {(node.node_id, node.type) for node in graph.nodes} == {
        (node.node_id, node.type) for node in round_trip.nodes
    }


def test_llm_converter_emits_prompts_and_tool_calls():
    graph = build_structural_graph("doc-123", 1, _sample_regions())
    prompts = graph_to_llm_prompts(graph)

    assert any("figure node image-1" in prompt for prompt in prompts)
    assert any("Relation has_caption" in prompt for prompt in prompts)

    calls = graph_to_tool_calls(graph)
    assert any(call["tool"] == "structural_relation" for call in calls)
    assert any(call["tool"] == "structural_node" for call in calls)
    figure_call = next(call for call in calls if call["arguments"]["id"] == "image-1")
    assert figure_call["arguments"]["bbox"]["width"] == 100
