"""Provider templates for downstream LLMs and helper vLMs.

The helpers intentionally avoid runtime dependencies so that users can
immediately drop their own endpoints, API keys, or local model paths into
one JSON file.  Both downstream LLMs and auxiliary vision-language models
share the same shape: each provider block advertises a ``provider`` label,
model identifier, auth fields, and an optional ``notes`` hint describing how
it is expected to be used.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

DOWNSTREAM_LLM_PRESETS: Dict[str, Dict[str, Any]] = {
    "local_hf": {
        "provider": "huggingface",
        "model_path": "/models/your-text-llm",  # e.g., local HF snapshot
        "revision": "main",
        "notes": "Point this to a local or network-mounted HF model for text-only downstream reasoning.",
    },
    "aws_bedrock": {
        "provider": "aws_bedrock",
        "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
        "region": "us-east-1",
        "profile": None,
        "endpoint_url": None,
        "api_key": "${AWS_BEDROCK_API_KEY}",
        "notes": "Set region/profile/endpoint to match your Bedrock deployment; API key required when STS is unavailable.",
    },
    "azure_openai": {
        "provider": "azure_openai",
        "endpoint": "https://your-resource.openai.azure.com/",
        "deployment": "gpt-4o-mini",
        "api_version": "2024-06-01",
        "api_key": "${AZURE_OPENAI_API_KEY}",
        "notes": "Use the deployment name you configured in Azure OpenAI and keep the api_version aligned with that region.",
    },
    "gemini": {
        "provider": "google_gemini",
        "model": "gemini-1.5-pro",
        "endpoint": "https://generativelanguage.googleapis.com/",
        "api_key": "${GOOGLE_API_KEY}",
        "notes": "Supply your Google API key; endpoint can stay default unless using a private gateway.",
    },
    "anthropic": {
        "provider": "anthropic",
        "model": "claude-3-5-sonnet-20241022",
        "api_key": "${ANTHROPIC_API_KEY}",
        "notes": "Pure text downstream reasoning via Anthropic's API.",
    },
}

AUXILIARY_VLM_PRESETS: Dict[str, Dict[str, Any]] = {
    "local_hf": {
        "provider": "huggingface",
        "model_path": "/models/your-vision-llm",  # e.g., local Llava/Vila
        "revision": "main",
        "notes": "Attach a local VLM checkpoint for visual enrichment calls.",
    },
    "aws_bedrock": {
        "provider": "aws_bedrock",
        "model_id": "anthropic.claude-3-opus-20240229-v1:0",
        "region": "us-east-1",
        "profile": None,
        "endpoint_url": None,
        "api_key": "${AWS_BEDROCK_API_KEY}",
        "notes": "Pick a Bedrock vision-capable model and ensure runtime has access to required IAM credentials.",
    },
    "azure_openai": {
        "provider": "azure_openai",
        "endpoint": "https://your-resource.openai.azure.com/",
        "deployment": "gpt-4o",
        "api_version": "2024-06-01",
        "api_key": "${AZURE_OPENAI_API_KEY}",
        "notes": "Use a multimodal Azure OpenAI deployment for screenshots or figure reasoning.",
    },
    "gemini": {
        "provider": "google_gemini",
        "model": "gemini-1.5-flash",  # faster vision-capable tier
        "endpoint": "https://generativelanguage.googleapis.com/",
        "api_key": "${GOOGLE_API_KEY}",
        "notes": "Good default for rapid visual side-car prompts; upgrade to Pro as needed.",
    },
    "anthropic": {
        "provider": "anthropic",
        "model": "claude-3-haiku-20240307",
        "api_key": "${ANTHROPIC_API_KEY}",
        "notes": "Vision-friendly Claude tier for lightweight inspections.",
    },
}


def render_provider_templates() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Return a merged provider template split by downstream role."""

    return {
        "downstream_llm": DOWNSTREAM_LLM_PRESETS,
        "helper_vlm": AUXILIARY_VLM_PRESETS,
    }


def write_provider_templates(path: str | Path) -> str:
    """Write a JSON file containing both LLM and VLM provider stubs.

    The output only contains placeholders and documented keys so callers can
    ``sed`` or edit in-place without reading source code.
    """

    dst = Path(path)
    payload = render_provider_templates()
    dst.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return str(dst)
