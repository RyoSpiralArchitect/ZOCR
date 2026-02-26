# Changelog

All notable changes to this project will be documented in this file.

The format is based on *Keep a Changelog*, and this project aims to follow *Semantic Versioning*.

## [Unreleased]

### Added
- `zocr.manifest.json` artifact manifest for run directories.
- `python -m zocr validate` CLI to validate manifests + core artifacts.
- Persistent job API endpoints (`/v1/jobs`) with storage rooted at `ZOCR_API_STORAGE_DIR`.
- `python -m zocr bench` lightweight benchmark harness.

### Changed
- Orchestrator now writes `zocr.manifest.json` and records `manifest_json` in `pipeline_summary.json`.

## [0.1.0] - 2026-02-25

### Added
- Initial alpha release of the Z-OCR suite (consensus + core + orchestrator + semantic diff).
