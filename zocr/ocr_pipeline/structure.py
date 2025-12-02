from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .models import (
    BoundingBox,
    RegionOutput,
    RegionType,
    StructuralEdge,
    StructuralGraph,
    StructuralNode,
)

__all__ = [
    "build_structural_graph",
    "graph_to_jsonld",
    "graph_from_jsonld",
    "save_jsonld",
    "load_jsonld",
    "graph_to_llm_prompts",
    "graph_to_tool_calls",
]


_UNIT_PATTERN = re.compile(r"\b([A-Za-z%][A-Za-z0-9%/μ]*)\b")


def _dedupe(seq: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _extract_units(header: str) -> List[str]:
    if not header:
        return []
    candidates = list(_UNIT_PATTERN.findall(header))
    paren_hits: List[str] = []
    for chunk in re.findall(r"\\(([^)]+)\\)", header):
        paren_hits.extend(_UNIT_PATTERN.findall(chunk))
    return _dedupe([c for c in candidates + paren_hits if c])


def _nearest_text_region(
    region: RegionOutput, text_regions: Sequence[RegionOutput]
) -> Optional[RegionOutput]:
    if not text_regions:
        return None
    if region.reading_order is None:
        return None
    scored: List[tuple[float, RegionOutput]] = []
    for text_region in text_regions:
        if text_region.reading_order is None:
            continue
        scored.append((abs(text_region.reading_order - region.reading_order), text_region))
    scored.sort(key=lambda item: item[0])
    return scored[0][1] if scored else None


def build_structural_graph(
    document_id: str, page_number: int, regions: Sequence[RegionOutput]
) -> StructuralGraph:
    nodes: List[StructuralNode] = []
    edges: List[StructuralEdge] = []

    text_regions = [region for region in regions if region.type == RegionType.TEXT]

    for region in regions:
        node_type = {
            RegionType.TEXT: "text_span",
            RegionType.IMAGE: "figure",
            RegionType.TABLE: "table",
        }.get(region.type, "region")
        attributes: Dict[str, Any] = {"reading_order": region.reading_order}
        content = region.content or {}

        if region.type == RegionType.TEXT:
            attributes["text"] = content.get("text")
        elif region.type == RegionType.IMAGE:
            caption = content.get("caption")
            if caption:
                caption_node = StructuralNode(
                    node_id=f"{region.region_id}#caption",
                    type="caption",
                    page_number=page_number,
                    attributes={"text": caption, "for_region": region.region_id},
                )
                nodes.append(caption_node)
                edges.append(
                    StructuralEdge(
                        source=region.region_id,
                        target=caption_node.node_id,
                        relation="has_caption",
                    )
                )
                edges.append(
                    StructuralEdge(
                        source=caption_node.node_id,
                        target=region.region_id,
                        relation="describes",
                    )
                )
            attributes["caption"] = caption
        elif region.type == RegionType.TABLE:
            table_data = content.get("table_data") or {}
            headers = table_data.get("headers") or []
            units = _dedupe([unit for header in headers for unit in _extract_units(str(header))])
            attributes.update({"headers": headers, "units": units})
            for idx, header in enumerate(headers):
                header_node = StructuralNode(
                    node_id=f"{region.region_id}#header{idx}",
                    type="table_header",
                    page_number=page_number,
                    attributes={
                        "text": header,
                        "units": _extract_units(str(header)),
                        "column_index": idx,
                    },
                )
                nodes.append(header_node)
                edges.append(
                    StructuralEdge(
                        source=region.region_id,
                        target=header_node.node_id,
                        relation="has_header",
                    )
                )

        nodes.append(
            StructuralNode(
                node_id=region.region_id,
                type=node_type,
                page_number=page_number,
                bounding_box=region.bounding_box,
                attributes=attributes,
            )
        )

        if region.type == RegionType.IMAGE:
            nearest_text = _nearest_text_region(region, text_regions)
            if nearest_text:
                edges.append(
                    StructuralEdge(
                        source=region.region_id,
                        target=nearest_text.region_id,
                        relation="near_text",
                    )
                )
                edges.append(
                    StructuralEdge(
                        source=nearest_text.region_id,
                        target=region.region_id,
                        relation="describes_figure",
                    )
                )

    return StructuralGraph(document_id=document_id, nodes=nodes, edges=edges)


def graph_to_jsonld(graph: StructuralGraph) -> Dict[str, Any]:
    context = {
        "@vocab": "https://zocr.ai/structure#",
        "bbox": "https://schema.org/box",
        "page": "https://schema.org/pageNumber",
        "relation": "https://schema.org/interactionType",
    }
    entries: List[Dict[str, Any]] = []
    for node in graph.nodes:
        payload: Dict[str, Any] = {
            "@id": node.node_id,
            "@type": node.type,
            "document": graph.document_id,
        }
        if node.page_number is not None:
            payload["page"] = node.page_number
        if node.bounding_box is not None:
            payload["bbox"] = node.bounding_box.model_dump()
        if node.attributes:
            payload["attributes"] = node.attributes
        entries.append(payload)

    for edge in graph.edges:
        entries.append(
            {
                "@id": f"{edge.source}->{edge.target}:{edge.relation}",
                "@type": "Relationship",
                "document": graph.document_id,
                "source": edge.source,
                "target": edge.target,
                "relation": edge.relation,
                "attributes": edge.attributes or {},
            }
        )

    return {"@context": context, "@id": f"urn:zocr:{graph.document_id}:structure", "@graph": entries}


def graph_from_jsonld(payload: Dict[str, Any]) -> StructuralGraph:
    document_id = payload.get("@id", "").replace("urn:zocr:", "").replace(":structure", "") or "doc"
    nodes: List[StructuralNode] = []
    edges: List[StructuralEdge] = []

    for entry in payload.get("@graph", []):
        if entry.get("@type") == "Relationship":
            edges.append(
                StructuralEdge(
                    source=entry.get("source"),
                    target=entry.get("target"),
                    relation=entry.get("relation", "related_to"),
                    attributes=entry.get("attributes") or {},
                )
            )
            continue
        bbox_data = entry.get("bbox")
        bbox = BoundingBox(**bbox_data) if isinstance(bbox_data, dict) else None
        nodes.append(
            StructuralNode(
                node_id=entry.get("@id"),
                type=entry.get("@type", "node"),
                page_number=entry.get("page"),
                bounding_box=bbox,
                attributes=entry.get("attributes") or {},
            )
        )

    return StructuralGraph(document_id=document_id, nodes=nodes, edges=edges)


def save_jsonld(graph: StructuralGraph, path: str) -> None:
    payload = graph_to_jsonld(graph)
    with open(path, "w", encoding="utf-8") as fw:
        json.dump(payload, fw, ensure_ascii=False, indent=2)


def load_jsonld(path: str) -> StructuralGraph:
    with open(path, "r", encoding="utf-8") as fr:
        payload = json.load(fr)
    return graph_from_jsonld(payload)


def graph_to_llm_prompts(graph: StructuralGraph) -> List[str]:
    prompts: List[str] = []
    for node in graph.nodes:
        parts = [f"{node.type} node {node.node_id}"]
        if node.page_number is not None:
            parts.append(f"page {node.page_number}")
        if node.bounding_box is not None:
            box = node.bounding_box
            parts.append(f"bbox x={box.x}, y={box.y}, w={box.width}, h={box.height}")
        if text := node.attributes.get("text"):
            parts.append(f"text '{text}'")
        if caption := node.attributes.get("caption"):
            parts.append(f"caption '{caption}'")
        if headers := node.attributes.get("headers"):
            parts.append(f"headers {headers}")
        if units := node.attributes.get("units"):
            parts.append(f"units {units}")
        prompts.append("; ".join(parts))

    for edge in graph.edges:
        prompts.append(f"Relation {edge.relation}: {edge.source} -> {edge.target}")
    return prompts


def graph_to_tool_calls(graph: StructuralGraph) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for node in graph.nodes:
        calls.append(
            {
                "tool": "structural_node",
                "arguments": {
                    "id": node.node_id,
                    "type": node.type,
                    "page": node.page_number,
                    "bbox": node.bounding_box.model_dump() if node.bounding_box else None,
                    "attributes": node.attributes,
                },
            }
        )
    for edge in graph.edges:
        calls.append(
            {
                "tool": "structural_relation",
                "arguments": {
                    "source": edge.source,
                    "target": edge.target,
                    "relation": edge.relation,
                    "attributes": edge.attributes,
                },
            }
        )
    return calls
