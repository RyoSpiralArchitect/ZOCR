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

- Local execution locks require a shared filesystem across API/worker processes;
  distributed execution uses PostgreSQL session advisory locks. Do not claim
  network-partition fencing or exactly-once execution from these locks alone.
- POSIX deadline enforcement kills the subprocess group; a child watchdog also
  stops orphaned work after supervisor death. Windows descendant cleanup and
  distributed database-failover fencing remain separate hardening work.
- Review HTML is a local evidence/decision-export workflow, not a server-side
  approval ledger. `publication_status` is a recommendation, not an access-control
  rule. Diagnostic artifacts remain downloadable.
- Preserve observed text and provenance. Generic augmentation must not fabricate
  invoice headers or overwrite weak OCR observations with inferred text.
