# Changelog

All notable changes to this project will be documented in this file.

The format is based on *Keep a Changelog*, and this project aims to follow *Semantic Versioning*.

## [Unreleased]

### Added
- `zocr.manifest.json` artifact manifest for run directories.
- `python -m zocr validate` CLI to validate manifests + core artifacts.
- Persistent job API endpoints (`/v1/jobs`) with storage rooted at `ZOCR_API_STORAGE_DIR`.
- `python -m zocr bench` lightweight benchmark harness.
- Docker CI job to build the image and run a minimal smoke check.
- `.env.example`, `compose.prod.yaml`, and `docs/DEPLOYMENT.md` for internal deployments.
- Internal delivery bundle scripts (`scripts/make_delivery.sh`, `scripts/verify_delivery.sh`) and `docs/DELIVERY.md`.
- Reference API observability: structured JSON logs, `X-Request-ID`, and Prometheus-style `/metrics`.
- Optional Kubernetes/Helm templates under `deploy/`.

### Changed
- Orchestrator now writes `zocr.manifest.json` and records `manifest_json` in `pipeline_summary.json`.
- Reference API now supports job retention/resume settings (`ZOCR_API_JOBS_*`) and uses FastAPI lifespan events.
- `compose.yaml` can now run from a pre-built image via `ZOCR_API_IMAGE` (avoids rebuilds for internal delivery).

## [0.1.1] - 2026-02-26

### Added
- Multi-arch delivery bundles via `ZOCR_DELIVERY_DOCKER_PLATFORMS` (one Docker image tar per platform).

### Changed
- Release workflow now builds/pushes multi-arch Docker images (`linux/amd64`, `linux/arm64`).
- Delivery script output now reliably lists bundled Docker image tags.

## [0.1.0] - 2026-02-25

### Added
- Initial alpha release of the Z-OCR suite (consensus + core + orchestrator + semantic diff).
