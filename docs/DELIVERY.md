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

## Run (Docker) / Dockerで使う
Load the image tar then start compose without building:
```bash
docker load -i zocr-suite-*-docker.tar
export ZOCR_API_IMAGE="zocr-suite:<version>"
docker compose up -d --no-build
curl http://127.0.0.1:8000/healthz
```

See `docs/DEPLOYMENT.md` for `.env` and hardening options.
