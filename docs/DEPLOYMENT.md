# Deployment (Internal) / 社内デプロイ

## TL;DR
```bash
cd ZOCR
cp .env.example .env
# (optional) edit .env
docker compose up -d --build
curl http://127.0.0.1:8000/healthz
```

Using a pre-built image (recommended for internal delivery bundles):
```bash
docker load -i /path/to/zocr-suite-*-docker.tar
export ZOCR_API_IMAGE="zocr-suite:<version>"
docker compose up -d --no-build
```

For a slightly hardened setup:
```bash
docker compose -f compose.yaml -f compose.prod.yaml up -d --build
```

## Data persistence / データ永続化
- Storage backend: `ZOCR_API_STORAGE_BACKEND` (default: `local`).
- Job artifacts are stored under `ZOCR_API_STORAGE_DIR` (default: `/data`).
- `compose.yaml` mounts a named volume to `/data` (`zocr_api_data`).
- Back up the volume if you need to retain historical results.

## Retention / 保存期間
Tune via `.env`:
- `ZOCR_API_JOBS_TTL_HOURS` (default 168 = 7 days): terminal jobs (`succeeded`/`failed`) older than TTL are deleted.
- `ZOCR_API_JOBS_MAX_COUNT` (default 200): keeps up to N terminal jobs (queued/running jobs are not deleted by max-count).

## Startup behavior / 起動時の挙動
- `ZOCR_API_JOBS_RESUME_ON_STARTUP=1` re-queues persisted `queued`/`running` jobs on startup.
- `ZOCR_API_JOBS_CLEANUP_ON_STARTUP=1` runs best-effort cleanup at startup.

## Observability / 監視
- Logs are emitted as JSON lines by default (`ZOCR_API_LOG_FORMAT=json`).
- Every HTTP response includes `X-Request-ID` (you can also supply it on the request).
- Prometheus-style metrics are exposed at `/metrics` when `ZOCR_API_METRICS_ENABLED=1`.
  - If `ZOCR_API_KEY` is set, `/metrics` also requires the same API key header.

## Performance knobs / 性能パラメータ
- `ZOCR_API_WORKERS`: uvicorn worker processes (total slots = workers × concurrency).
- `ZOCR_API_CONCURRENCY`: parallel pipeline slots inside one process.
- `ZOCR_API_ZIP_COMPRESSION=stored` can make `artifacts.zip` downloads faster to generate
  (at the cost of larger files); keep `deflated` for smaller bundles.

## Security notes / セキュリティ注意
- Set `ZOCR_API_KEY` and place the service behind your internal reverse proxy / network controls.
- Treat the reference API as an internal wrapper; production hardening (rate limit, mTLS, audit logging) should be added per environment.

## Kubernetes / Helm (Optional) / Kubernetes/Helm（任意）
- Kustomize: `deploy/k8s/`
- Helm chart: `deploy/helm/zocr-suite/`
