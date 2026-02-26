# Compliance Artifacts (Draft) / コンプライアンス成果物（ドラフト）

This document describes **practical** steps to generate basic compliance artifacts
for commercial/on-prem and SaaS deployments. It is **not legal advice**.

## Recommended release artifacts

- `THIRD_PARTY_NOTICES.md` (Python dependency inventory; best-effort)
- `sbom-python.cdx.json` (CycloneDX SBOM for the Python environment)
- `sbom-image.*.json` (SBOM for the Docker image, including OS packages)
- Optional: vulnerability scan reports (dependency + container)

## Generate (Python)

This creates:
- `compliance/THIRD_PARTY_NOTICES.md`
- `compliance/sbom-python.cdx.json`
- `compliance/requirements.lock.txt`

```bash
cd ZOCR
bash scripts/generate_compliance_artifacts.sh
```

Options:
- `ZOCR_COMPLIANCE_OUTDIR=compliance`
- `ZOCR_COMPLIANCE_EXTRAS=api` (comma-separated extras like `api,pdf`)
- `ZOCR_COMPLIANCE_GATHER_LICENSE_TEXTS=1` (larger SBOM; includes license texts when available)

## Generate (Docker image)

The recommended path is to generate an SBOM from the built image in CI using a
tool like Syft (e.g. via `anchore/sbom-action`) and attach it to releases.

