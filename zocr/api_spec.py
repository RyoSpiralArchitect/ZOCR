"""API contract and prompt templates for the Z-OCR ingestion/query surface.

This module captures the minimal JSON Schemas and prompt templates outlined for
Phase 4 so downstream services (HTTP handlers, SDKs, or tests) can import a
single source of truth without relying on ad-hoc Markdown. The schemas follow
Draft 2020-12 and intentionally keep `additionalProperties` disabled to
highlight the canonical v0 shape.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

__all__ = [
    "INGEST_REQUEST_SCHEMA_V0",
    "INGEST_RESPONSE_SCHEMA_V0",
    "QUERY_REQUEST_SCHEMA_V0",
    "QUERY_RESPONSE_SCHEMA_V0",
    "SYSTEM_PROMPT_ANALYSIS_V0",
    "USER_PROMPT_ANALYSIS_TEMPLATE_V0",
    "get_api_schemas_v0",
    "get_prompt_templates_v0",
]

INGEST_REQUEST_SCHEMA_V0: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "IngestRequest",
    "type": "object",
    "additionalProperties": False,
    "required": ["tenant_id", "files"],
    "properties": {
        "tenant_id": {
            "type": "string",
            "minLength": 1,
            "description": "Logical tenant / customer identifier",
        },
        "files": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "uri"],
                "properties": {
                    "id": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Client-side identifier for this file",
                    },
                    "uri": {
                        "type": "string",
                        "format": "uri",
                        "description": "Location of the uploaded file (e.g. s3://..., https://...)",
                    },
                    "kind": {
                        "type": "string",
                        "enum": ["auto", "pdf", "image"],
                        "default": "auto",
                    },
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "domain_hint": {
            "type": "string",
            "description": "Optional domain hint such as 'invoice', 'bank_statement', or 'auto'",
            "default": "auto",
        },
        "options": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "snapshot": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to enable snapshot / episode artifacts",
                },
                "seed": {"type": "integer", "description": "Optional RNG seed"},
                "priority": {
                    "type": "string",
                    "enum": ["low", "normal", "high"],
                    "default": "normal",
                },
            },
        },
    },
}

INGEST_RESPONSE_SCHEMA_V0: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "IngestResponse",
    "type": "object",
    "additionalProperties": False,
    "required": ["job_id", "tenant_id", "status"],
    "properties": {
        "job_id": {"type": "string", "minLength": 1},
        "tenant_id": {"type": "string", "minLength": 1},
        "status": {"type": "string", "enum": ["queued", "running", "completed", "failed"]},
        "created_at": {"type": "string", "format": "date-time"},
        "estimated_pages": {"type": "integer", "minimum": 0},
        "error": {"type": "string"},
        "summary": {
            "type": "object",
            "additionalProperties": True,
            "description": "Optional summary (page counts, domains, artifact paths, etc.)",
        },
    },
}

QUERY_REQUEST_SCHEMA_V0: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "QueryRequest",
    "type": "object",
    "additionalProperties": False,
    "required": ["tenant_id", "job_id", "query"],
    "properties": {
        "tenant_id": {"type": "string", "minLength": 1},
        "job_id": {"type": "string", "minLength": 1},
        "conversation_id": {
            "type": "string",
            "description": "Optional conversation/thread identifier",
        },
        "query": {
            "type": "string",
            "minLength": 1,
            "description": "Natural-language question from the end user",
        },
        "language": {
            "type": "string",
            "description": "Preferred output language (BCP-47)",
            "default": "ja",
            "pattern": "^[a-z]{2}(-[A-Z]{2})?$",
        },
        "mode": {
            "type": "string",
            "enum": ["analysis", "qa", "explain"],
            "default": "analysis",
        },
    },
}

QUERY_RESPONSE_SCHEMA_V0: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "QueryResponse",
    "type": "object",
    "additionalProperties": False,
    "required": ["answer"],
    "properties": {
        "answer": {
            "type": "object",
            "additionalProperties": False,
            "required": ["data_summary", "business_commentary"],
            "properties": {
                "data_summary": {
                    "type": "string",
                    "description": "Grounded summary based only on retrieved facts",
                },
                "business_commentary": {
                    "type": "string",
                    "description": "General business / management commentary",
                },
            },
        },
        "artifacts": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "tables": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["id", "columns", "rows"],
                        "properties": {
                            "id": {"type": "string"},
                            "title": {"type": "string"},
                            "columns": {
                                "type": "array",
                                "minItems": 1,
                                "items": {"type": "string"},
                            },
                            "rows": {"type": "array", "items": {"type": "array", "items": {}}},
                        },
                    },
                    "default": [],
                },
                "charts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["id", "type", "spec"],
                        "properties": {
                            "id": {"type": "string"},
                            "type": {"type": "string"},
                            "spec": {"type": "object"},
                        },
                    },
                    "default": [],
                },
            },
            "default": {"tables": [], "charts": []},
        },
        "provenance": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["trace"],
                "properties": {
                    "trace": {"type": "string"},
                    "fact_text": {"type": "string"},
                    "table_id": {"type": "string"},
                },
            },
            "default": [],
        },
        "flags": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "facts_insufficient": {"type": "boolean", "default": False},
            },
            "default": {},
        },
    },
}

SYSTEM_PROMPT_ANALYSIS_V0 = """You are a data analyst assisting INDX customers with structured financial data.

You receive:
- A QUESTION written by the end user.
- A FACTS section containing only verified facts extracted from documents
  (tables, metrics, entities) that you can trust.
- Optional TABLES rendered in compact form for your convenience.
- Optional FLAGS summarising whether the retrieved facts are sufficient.

Your goals:
1. For any specific number, date, amount, rate, or named entity,
   you MUST rely only on the given FACTS/TABLES. Never invent document-specific facts.
2. If the facts are insufficient to answer precisely, explicitly say so and avoid
   fabricating numbers. You may still provide general, non-document-specific advice.
3. Produce two clearly separated sections in your answer:
   - "Data-grounded summary": what can be concluded strictly from the given facts.
   - "Business commentary": general patterns, risks, or opportunities. This section
     may use your broader knowledge but must be phrased as general guidance
     (e.g. "一般的には…", "一般的な傾向として…").

Output format (Markdown, in Japanese unless the QUESTION requests another language):

### Data-grounded summary
- …

### Business commentary
- …

Do not mention internal IDs or trace strings in your answer.
If there is a conflict between the QUESTION and the FACTS, trust the FACTS.
Keep the answer concise and focused on the user's QUESTION.
"""

USER_PROMPT_ANALYSIS_TEMPLATE_V0 = """QUESTION:
{{ user_query }}

FACTS:
{{#each facts_as_sentences}}
- [trace={{trace}}] {{text}}
{{/each}}

TABLES:
{{#each tables}}
Table: {{title}}
{{markdown_table_here}}
{{/each}}

FLAGS:
- facts_insufficient: {{true_or_false}}
"""


def get_api_schemas_v0() -> Dict[str, Dict[str, Any]]:
    """Return deep copies of the canonical v0 JSON Schemas."""

    return {
        "ingest_request": deepcopy(INGEST_REQUEST_SCHEMA_V0),
        "ingest_response": deepcopy(INGEST_RESPONSE_SCHEMA_V0),
        "query_request": deepcopy(QUERY_REQUEST_SCHEMA_V0),
        "query_response": deepcopy(QUERY_RESPONSE_SCHEMA_V0),
    }


def get_prompt_templates_v0() -> Dict[str, str]:
    """Return deep copies of the prompt templates for analysis mode."""

    return {
        "system_analysis": str(SYSTEM_PROMPT_ANALYSIS_V0),
        "user_analysis": str(USER_PROMPT_ANALYSIS_TEMPLATE_V0),
    }
