# Deployment (Internal) / 社内デプロイ

## TL;DR
On-prem / single-node:
```bash
cd ZOCR
cp .env.example .env
# (optional) edit .env
docker compose up -d --build
curl http://127.0.0.1:8000/healthz
```

SaaS / multi-worker baseline (Redis queue + Postgres metadata + S3/MinIO storage):
```bash
cd ZOCR
cp .env.example .env
# edit .env:
#   ZOCR_EXTRAS=api,api_s3,api_redis,api_db
#   ZOCR_API_METADATA_BACKEND=postgres
#   ZOCR_API_STORAGE_BACKEND=s3
#   ZOCR_API_QUEUE_BACKEND=redis
#   ZOCR_API_DATABASE_URL=postgresql://zocr:zocr@postgres:5432/zocr
#   ZOCR_API_S3_BUCKET=...
docker compose --profile saas up -d --build
curl http://127.0.0.1:8000/healthz
```

Using a pre-built image (recommended for internal delivery bundles):
```bash
docker load -i /path/to/zocr-suite-*-docker.tar  # pick the tar matching your host arch (amd64/arm64)
export ZOCR_API_IMAGE="zocr-suite:<version>"
docker compose up -d --no-build
```

For a slightly hardened setup:
```bash
docker compose -f compose.yaml -f compose.prod.yaml up -d --build
```

## Data persistence / データ永続化
- Storage backend: `ZOCR_API_STORAGE_BACKEND` (default: `local`).
- Metadata backend: `ZOCR_API_METADATA_BACKEND` (default: `local`).
- `local`: job metadata is stored together with job artifacts under `ZOCR_API_STORAGE_DIR` (default: `/data`).
- `postgres`: job metadata lives in Postgres, while uploads/artifacts still use the configured storage backend.
- `s3`: uploads/artifacts are persisted to S3-compatible object storage, while `ZOCR_API_STORAGE_DIR`
  remains a local staging cache on the API/worker nodes.
- `compose.yaml` mounts a named volume to `/data` (`zocr_api_data`).
- If you run `local` storage with Redis workers, API and worker nodes must share the same filesystem/volume.

## Postgres metadata / メタデータDB
Required when `ZOCR_API_METADATA_BACKEND=postgres`:
- `ZOCR_API_DATABASE_URL`
- `ZOCR_API_DB_AUTO_INIT=1` auto-creates the reference table/indexes on startup.
- `compose.yaml` exposes `postgres` under the `saas` profile and wires API/worker to it by default.
- The reference schema stores one JSON payload per `(tenant_id, job_id)` plus indexed status/timestamps.
- It also creates `zocr_tenant_plans` / `zocr_tenant_policies`, so quota / rate-limit plans can be managed in DB instead of only env vars.

## Retention / 保存期間
Tune via `.env`:
- `ZOCR_API_JOBS_TTL_HOURS` (default 168 = 7 days): terminal jobs (`succeeded`/`failed`) older than TTL are deleted.
- `ZOCR_API_JOBS_MAX_COUNT` (default 200): keeps up to N terminal jobs per tenant (queued/running jobs are not deleted by max-count).
- `ZOCR_API_TENANT_MAX_ACTIVE_JOBS` (default 0 = disabled): rejects new creates when a tenant already has N `queued`/`running` jobs.
- `ZOCR_API_TENANT_MAX_STORED_JOBS` (default 0 = disabled): rejects new creates when a tenant already has N persisted jobs.

## Startup behavior / 起動時の挙動
- `ZOCR_API_JOBS_RESUME_ON_STARTUP=1` re-queues persisted `queued`/`running` jobs on startup.
- `ZOCR_API_JOBS_CLEANUP_ON_STARTUP=1` runs best-effort cleanup at startup.

## Queue backends / ジョブ実行バックエンド
- `ZOCR_API_QUEUE_BACKEND=inline` (default): API process executes jobs itself. Best for on-prem / single-node.
- `ZOCR_API_QUEUE_BACKEND=redis`: API enqueues jobs into Redis Streams, and `zocr-worker` executes them.
- `compose.yaml` exposes `worker`, `redis`, and `postgres` under the `saas` profile:
  ```bash
  docker compose --profile saas up -d --build
  ```
- Workers reclaim stale pending Redis messages after `ZOCR_WORKER_CLAIM_IDLE_MS` (default 60000 ms),
  which helps recover jobs after a worker crash.

## S3 / MinIO storage / オブジェクトストレージ
Required when `ZOCR_API_STORAGE_BACKEND=s3`:
- `ZOCR_API_S3_BUCKET`
- `ZOCR_API_S3_PREFIX` (default `zocr`)
- `ZOCR_API_S3_REGION` (optional for AWS)
- `ZOCR_API_S3_ENDPOINT_URL` (use for MinIO / non-AWS S3)
- `ZOCR_API_S3_FORCE_PATH_STYLE=1` (often needed for MinIO)
- `ZOCR_API_S3_PRESIGN_ENABLED=1` enables redirect-based downloads for artifacts/zip.
- `ZOCR_API_S3_PRESIGN_TTL_SEC` controls signed URL lifetime.

## Tenant scoping / テナント分離
- `ZOCR_API_MULTI_TENANT_ENABLED=1` enables tenant-aware job scoping in the reference API.
- Set `ZOCR_API_DEFAULT_TENANT=` (blank) to require `X-Tenant-ID` on every request.
- Job list/get/delete/artifact routes are scoped by tenant and return `404` for cross-tenant access.
- For single-tenant/on-prem operation, leave `ZOCR_API_MULTI_TENANT_ENABLED=0` and use the default tenant.
- Env-managed plans are also supported for local/dev use:
  - `ZOCR_API_TENANT_PLANS_JSON={"starter":{"rate_limit_per_min":60,"max_active_jobs":2,"max_stored_jobs":50}}`
  - `ZOCR_API_TENANT_POLICIES_JSON={"tenant-a":{"plan":"starter"}}`
- When metadata backend is `postgres`, tenant policy resolution comes from `zocr_tenant_policies` + optional `zocr_tenant_plans`.

## Tenant admin / テナント管理
- Principals with `zocr.tenants.read` can list plans/policies, and `zocr.tenants.write` can mutate them:
  ```bash
  curl -X PUT http://127.0.0.1:8000/v1/admin/tenant-plans/starter \
    -H "X-API-Key: $ZOCR_ADMIN_KEY" \
    -H "Content-Type: application/json" \
    -d '{"max_active_jobs":2,"rate_limit_per_min":60}'
  curl -X PUT http://127.0.0.1:8000/v1/admin/tenant-policies/tenant-a \
    -H "X-API-Key: $ZOCR_ADMIN_KEY" \
    -H "Content-Type: application/json" \
    -d '{"plan_name":"starter"}'
  ```
- Set `ZOCR_API_TENANT_APPROVAL_REQUIRED=1` to turn those write APIs into request creation. They return `202 Accepted`, and a reviewer with `zocr.tenants.approve` must approve/reject the pending request:
  ```bash
  curl -X GET "http://127.0.0.1:8000/v1/admin/tenant-change-requests?status=pending" \
    -H "X-API-Key: $ZOCR_APPROVER_KEY"
  curl -X POST http://127.0.0.1:8000/v1/admin/tenant-change-requests/$REQUEST_ID/approve \
    -H "X-API-Key: $ZOCR_APPROVER_KEY" \
    -H "Content-Type: application/json" \
    -d '{"reason":"approved"}'
  ```
- By default, the original requester cannot approve their own change request. Set `ZOCR_API_TENANT_APPROVAL_ALLOW_SELF=1` only if you intentionally want to relax that separation of duties.
- `ZOCR_API_TENANT_APPROVAL_MIN_APPROVERS=2` keeps the request in `pending` after the first approval and emits `tenant_change_request.reviewed`; the final approver flips it to `approved` and applies the mutation.
- `ZOCR_API_TENANT_APPROVAL_NOTIFY_URL` sends `tenant_change_request.created|reviewed|approved|rejected` webhooks for both API and `zocr-admin` mutations. Optional headers/timeout come from `ZOCR_API_TENANT_APPROVAL_NOTIFY_HEADERS_JSON` and `ZOCR_API_TENANT_APPROVAL_NOTIFY_TIMEOUT_SEC`.
- The same mutations are available locally via `zocr-admin`:
  ```bash
  zocr-admin tenant-plan put starter --max-active-jobs 2 --rate-limit-per-min 60
  zocr-admin tenant-policy put tenant-a --plan starter
  zocr-admin tenant-plan list
  zocr-admin tenant-policy list
  ```
- With approval mode enabled, use `zocr-admin tenant-request list|approve|reject` to review pending changes:
  ```bash
  zocr-admin tenant-request list --status pending
  zocr-admin tenant-request approve "$REQUEST_ID" --reason approved
  ```
- Audit events can be searched via API / CLI:
  ```bash
  curl "http://127.0.0.1:8000/v1/admin/audit-events?tenant_id=tenant-a&event=tenant_policy.upserted&limit=20" \
    -H "X-API-Key: $ZOCR_ADMIN_KEY"
  zocr-admin audit list --tenant-id tenant-a --event tenant_policy.upserted --limit 20
  ```

## Observability / 監視
- Logs are emitted as JSON lines by default (`ZOCR_API_LOG_FORMAT=json`).
- Every HTTP response includes `X-Request-ID` (you can also supply it on the request).
- `ZOCR_API_AUDIT_SINKS=file,postgres,http` enables one or more audit sinks.
- `ZOCR_API_AUDIT_READ_BACKEND=auto|file|postgres` selects which stored sink backs audit search APIs (`auto` prefers Postgres, then file).
  - `file`: append-only JSONL via `ZOCR_API_AUDIT_LOG_PATH`
  - `postgres`: stores audit events in `zocr_audit_events` via `ZOCR_API_AUDIT_DATABASE_URL` (or `ZOCR_API_DATABASE_URL`)
  - `http`: pushes JSON events to a SIEM/webhook via `ZOCR_API_AUDIT_HTTP_URL`
- The same sinks also receive `zocr-admin` plan/policy mutations and approval workflow events.
- Prometheus-style metrics are exposed at `/metrics` when `ZOCR_API_METRICS_ENABLED=1`.
  - If auth is enabled, `/metrics` requires the configured API key or JWT.

## Auth / 認証
- `ZOCR_API_AUTH_MODE=auto` (default) auto-detects `none`, `api_key`, `jwt`, or `hybrid`.
- `ZOCR_API_AUTHZ_STRICT=1` switches scoped routes to deny unscoped principals.
- API key mode:
  - `ZOCR_API_KEY` + `ZOCR_API_KEY_TENANT` for a single shared key
  - optional `ZOCR_API_KEY_ROLES` / `ZOCR_API_KEY_SCOPES` for the legacy single-key path
  - or `ZOCR_API_KEYS_JSON` for per-tenant / admin keys with optional `roles` / `scopes`
- JWT mode:
  - `ZOCR_API_JWT_SECRET`
  - or JWKS / OIDC settings: `ZOCR_API_JWT_JWKS_URL`, `ZOCR_API_JWT_JWKS_PATH`, `ZOCR_API_JWT_JWKS_JSON`, `ZOCR_API_OIDC_DISCOVERY_URL`
  - optional issuer/audience/claim envs (`ZOCR_API_JWT_*`), including `roles` / `scope` claims
  - for typical OIDC providers, set `ZOCR_API_JWT_ALGORITHMS=RS256`
- In `hybrid`, `X-API-Key` uses API-key auth and `Authorization: Bearer ...` uses JWT.
- For non-admin principals, tenant binding comes from the API key config or JWT claim; `X-Tenant-ID` cannot override it.
- Built-in roles:
  - `viewer` → `zocr.jobs.read`, `zocr.metrics.read`
  - `operator` → viewer + `zocr.jobs.write`, `zocr.jobs.run`
  - `admin` → operator + `zocr.audit.read`, `zocr.jobs.delete`, `zocr.tenants.read`, `zocr.tenants.write`, `zocr.tenants.approve`, `zocr.tenants.override`
- If a principal has no explicit roles/scopes, the reference API currently allows scoped routes for backward compatibility.
- Enable `ZOCR_API_AUTHZ_STRICT=1` to remove that compatibility path before exposing the API to external tenants.
- `ZOCR_API_OIDC_DISCOVERY_URL` automatically resolves `jwks_uri`; if `ZOCR_API_JWT_ISSUER` is unset, the discovered issuer is also enforced.

## Performance knobs / 性能パラメータ
- `ZOCR_API_WORKERS`: uvicorn worker processes (total slots = workers × concurrency).
- `ZOCR_API_CONCURRENCY`: parallel pipeline slots inside one process.
- `ZOCR_API_RATE_LIMIT_PER_MIN`: per principal+tenant request ceiling across scoped API routes (`0` disables).
- `ZOCR_API_RATE_LIMIT_BACKEND=local|redis`: `redis` makes limits shared across API replicas.
- `ZOCR_API_RATE_LIMIT_REDIS_URL`: optional override; defaults to `ZOCR_API_REDIS_URL`.
- `ZOCR_API_RATE_LIMIT_REDIS_PREFIX`: Redis key prefix for distributed limit buckets.
- Tenant plans can override `rate_limit_per_min`, `max_active_jobs`, and `max_stored_jobs` per tenant.
- `ZOCR_WORKER_BLOCK_MS`: Redis worker poll block time.
- `ZOCR_WORKER_CLAIM_IDLE_MS`: idle time before a worker re-claims a stuck Redis message.
- `ZOCR_API_ZIP_COMPRESSION=stored` can make `artifacts.zip` downloads faster to generate
  (at the cost of larger files); keep `deflated` for smaller bundles.

## Benchmarking / ベンチ
Run a quick client-side load test against the reference API:

```bash
# (example) benchmark /v1/run with a generated PNG + toy-lite pipeline
python -m zocr bench api --url http://127.0.0.1:8000 --requests 50 --concurrency 8 --toy-lite
```

If `ZOCR_API_KEY` is enabled, pass the same key:
```bash
python -m zocr bench api --url http://127.0.0.1:8000 --api-key "$ZOCR_API_KEY" --requests 50 --concurrency 8
```

To benchmark with a real document, pass `--file` (pdf/png/jpg/tiff) and optionally disable toy mode:
```bash
python -m zocr bench api --file path/to/sample.pdf --no-toy-lite --requests 10 --concurrency 2
```

## Security notes / セキュリティ注意
- Set `ZOCR_API_AUTH_MODE` and run the API behind your reverse proxy / network controls.
- For S3-backed downloads, review your bucket policy together with `ZOCR_API_S3_PRESIGN_ENABLED`.
- Prefer deriving tenant identity from API keys or JWT claims; reserve `X-Tenant-ID` for admin/service contexts only.
- Baseline rate limiting / tenant quotas / audit logging are built in, but mTLS, WAF, and central log shipping still belong in your environment layer.

## Kubernetes / Helm (Optional) / Kubernetes/Helm（任意）
- Kustomize: `deploy/k8s/`
- Helm chart: `deploy/helm/zocr-suite/`
