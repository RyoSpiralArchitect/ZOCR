from pathlib import Path

from zocr.resources.model_provider_presets import (
    AUXILIARY_VLM_PRESETS,
    DOWNSTREAM_LLM_PRESETS,
    render_provider_templates,
    write_provider_templates,
)


def test_templates_have_expected_providers():
    llm = set(DOWNSTREAM_LLM_PRESETS)
    vlm = set(AUXILIARY_VLM_PRESETS)
    assert {
        "local_hf",
        "aws_bedrock",
        "azure_openai",
        "gemini",
        "anthropic",
        "openai",
        "mistral",
        "xai",
    }.issubset(llm)
    assert {
        "local_hf",
        "aws_bedrock",
        "azure_openai",
        "gemini",
        "anthropic",
        "openai",
        "mistral",
        "xai",
    }.issubset(vlm)


def test_render_provider_templates_merges():
    payload = render_provider_templates()
    assert payload["downstream_llm"]["azure_openai"]["deployment"] == "gpt-4o-mini"
    assert payload["helper_vlm"]["azure_openai"]["deployment"] == "gpt-4o"


def test_write_provider_templates(tmp_path: Path):
    out = tmp_path / "providers.json"
    path = write_provider_templates(out)
    assert Path(path).exists()
    text = Path(path).read_text(encoding="utf-8")
    assert "downstream_llm" in text
    assert "helper_vlm" in text
