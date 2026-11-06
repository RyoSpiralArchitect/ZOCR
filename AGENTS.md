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
- Supported image extensions are shared in `zocr/utils/formats.py`; directory resume
  fingerprints must cover every accepted input type, including BMP and WebP.
- Review table matching must be deterministic across Python hash seeds and keep
  documents separate. Renamed single-document version pairs may match by headers;
  unrelated multi-document identities require explicit correspondence.
- Exported page bitmaps are archived under `evidence/` and included in manifests.
  RAG image references are relative to the run root, not temporary input paths.
- Final quality diagnostics bind the contextual, augmented and RAG cell hashes in
  `final_export.quality.json`; missing/stale bindings must not certify publication.
- Exercise S3 with both PostgreSQL and local metadata. Local review coordination
  still requires shared filesystem locks, even though ledger data persists in S3.

# Toy glyph learning trials

- Run `zocr-consensus toy-learn --dataset glyphs.jsonl --allowed-chars 0123456789 --out new-trial`.
  The output directory must not already exist. Trials run in a fresh interpreter,
  never modify the calling runtime and never auto-activate production memory.
- Each JSONL record requires `id`, `image`, single-character `text`, `group_id`,
  `split` (`train` or `holdout`), `approved: true`, `reviewer`, and `reason`.
  Images must be inside the dataset directory. Training and holdout document groups
  must be disjoint; repeated source or normalized glyph pixels are rejected.
- Holdout data needs at least two document groups. It is selection data, not an
  independent final test. Approval fields are asserted by the local dataset owner,
  unlike authenticated server review actors. Do not claim page-level OCR accuracy.
- `learning_report.json` records exact glyph accuracy, error rate, confidence
  diagnostics, latency, dataset/model/runtime hashes and per-sample results.
  `candidate_memory.json` is emitted only for strict improvement with zero
  individual holdout regressions. Both artifacts contain sensitive information.
- Test with `python -m pytest -q -s tests/test_toy_learning.py`. Timing diagnostics
  are not CI speed thresholds. Cached preparation must preserve exact results;
  its 128 entries are pixel-keyed, read-only and restricted to small templates.

# Approved learning and independent cell evaluation

- `zocr-consensus cell-eval --dataset cells.jsonl --out new-evaluation` measures raw
  full-cell CER/WER, retaining blank-reference insertions. WER uses whitespace,
  not Japanese linguistic segmentation. Each cell starts a fresh bounded process.
- Cell JSONL adds `document`, `group_id`, `source_kind` (`fixture` or `real_document`),
  and `split: test` to the approved image/text/reviewer/reason fields. File bytes and
  document-family groups must not overlap the model's training or selection lineage.
  A model without complete document lineage cannot certify an independent test.
- Learning sample registration requires `zocr.reviews.approve`, explicit consent,
  a current corrected review event, a document-family group, and an immutable split.
  Train/selection samples require manually labelled glyph boxes in cell coordinates;
  these spell the non-whitespace correction. No automatic character labelling is used.
- Learning models/evaluations additionally require `zocr.jobs.run`. Objects and
  document split assignments are tenant-local and immutable. Review revision changes
  invalidate samples and derived models. Evaluations are persisted, not just exported.
- Source input and archived page hashes must match the committed manifest; legacy
  jobs without hashed original inputs must be rerun before learning registration.
- PostgreSQL uses unique learning-object rows; S3/local metadata still needs shared
  filesystem coordination. Candidates never auto-activate. Source-kind declarations
  and document-family assignments are operator assertions, not proof of real-world
  provenance. Fixture results must never be reported as real-document performance.
