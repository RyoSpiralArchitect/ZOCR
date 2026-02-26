# Delivery Pack (Internal) / 納品パック（社内向け）

This repo can be delivered as a single “bundle” directory containing:
- Python packages (`dist/*.whl`, `dist/*.tar.gz`)
- Docker image tar (`docker save`)
- `SHA256SUMS` (integrity checks)
- Minimal runbook + compose files
- Optional Kubernetes/Helm templates (`deploy/`)

## Build a bundle / バンドル作成
```bash
cd ZOCR
bash scripts/make_delivery.sh
```

Outputs land in `delivery/` (timestamped subfolder).

If Docker is unavailable on the build machine, you can still produce a Python-only bundle:
```bash
ZOCR_DELIVERY_SKIP_DOCKER=1 bash scripts/make_delivery.sh
```

Optional (customize the Docker image build):
```bash
ZOCR_DELIVERY_DOCKER_EXTRAS=api,ocr_tess \
ZOCR_DELIVERY_DOCKER_APT_PACKAGES="poppler-utils tesseract-ocr tesseract-ocr-eng" \
  bash scripts/make_delivery.sh
```

Optional (recommended for commercial deliveries):
```bash
# Include SBOM + third-party notices under <bundle>/compliance/
ZOCR_DELIVERY_COMPLIANCE=1 bash scripts/make_delivery.sh
```

Optional (offline installs):
```bash
# Download wheels into <bundle>/wheels/ (build on the target OS/arch for best results)
ZOCR_DELIVERY_WHEELS=1 ZOCR_DELIVERY_WHEEL_EXTRAS=api bash scripts/make_delivery.sh
```

## Verify a bundle / 検証
```bash
bash scripts/verify_delivery.sh delivery/zocr-suite-*/SHA256SUMS
```

## Install (Python) / Pythonで使う
If your target environment has access to PyPI (or an internal mirror):
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install zocr_suite-*.whl
python -m zocr --version
```

If the environment is fully offline, you must also pre-stage wheels for runtime deps
(`numpy`, `Pillow`, and optionally FastAPI extras). A common pattern is to use an
internal PyPI mirror or attach a `wheels/` folder to the delivery.

If your bundle includes a `wheels/` directory, you can install without PyPI:
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --no-index --find-links wheels zocr_suite-*.whl
```

## Run (Docker) / Dockerで使う
Load the image tar then start compose without building:
```bash
docker load -i zocr-suite-*-docker.tar
export ZOCR_API_IMAGE="zocr-suite:<version>"
docker compose up -d --no-build
curl http://127.0.0.1:8000/healthz
```

See `docs/DEPLOYMENT.md` for `.env` and hardening options.
