# Verification

- Run `python -m pytest -q -rs`, `python -m ruff check .`, and `python -m build`.
- Python 3.12 is a tested full-dependency environment. The PDF ingestion tests need
  `multimodal` and `parquet` extras as well as the dev dependencies. The API tests
  need `api`; service integration tests also need `api_saas`.
- Tests normally skip external-service checks. To exercise dedicated local test
  services, set `ZOCR_TEST_REDIS_URL`, `ZOCR_TEST_POSTGRES_URL`, and
  `ZOCR_TEST_S3_URL`. These tests reject non-loopback endpoints and create uniquely
  named test streams, jobs, and buckets. Never point them at production services.
- For the optional browser test, install Playwright and Chromium, then set
  `ZOCR_TEST_BROWSER=1`. Use `PLAYWRIGHT_SKIP_BROWSER_GC=1` when installing browsers
  to avoid removing browser caches used by other projects.
- `tests/test_operational_pipeline.py` runs real bounded OCR subprocesses and an
  inline API job. Do not replace these with mocks to hide execution failures.
- Artifact manifests must be generated after the final summary is written. Do not
  rewrite a hashed artifact afterward. Service jobs validate required artifacts
  before persisting a successful status; optional-output warnings are not failures.

# Execution and review boundaries

- Local execution locks require a shared filesystem across API/worker processes.
  Distributed execution requires one authoritative PostgreSQL primary: guarded
  updates use the live lock session plus an execution token. This does not solve
  database split-brain or provide exactly-once computation.
- Each attempt writes its own immutable output namespace. Artifact endpoints only
  serve the successful attempt selected by committed metadata, never a running
  attempt or an on-demand reconstruction of an unpublished result.
- POSIX execution kills process groups and watches supervisor death. Windows uses
  a kill-on-close Job Object and assigns suspended children before resuming them.
  The windows-execution CI job exercises native deadline and supervisor cleanup.
- Review HTML still exports local decisions. The server review API binds source
  manifests/cell hashes, authenticated actors, revision checks and hash-chained
  decision history. Corrections are sidecars, not edits to source artifacts.
  This is an application-level ledger, not WORM storage or protection from a
  privileged administrator rewriting both data and hashes. Publication quality
  recommendations remain separate from human review and access control.
- `ZOCR_TEST_DOCKER_FAILOVER=1` runs dedicated streaming replication, primary stop
  and standby promotion. It stops only containers it created and retains them.
- `ZOCR_TEST_MINIO_URL` enables multipart/presign/cache/authentication checks against
  dedicated loopback MinIO. The test image pins official MinIO source commit
  `0d7408fc9969caf07de6a8c3a84f9fbb10a6739e` in `tests/fixtures/minio.Dockerfile`.
  AWS deployment validation is intentionally not part of these local tests.
- Preserve observed text and provenance. Generic augmentation must not fabricate
  invoice headers or overwrite weak OCR observations with inferred text.
