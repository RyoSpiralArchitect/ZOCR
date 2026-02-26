#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VERSION="$(python3 -S -c 'from zocr._version import __version__; print(__version__)')"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

OUT_ROOT="${ZOCR_DELIVERY_DIR:-delivery}"
BUNDLE_DIR="${OUT_ROOT}/zocr-suite-${VERSION}-${STAMP}"

mkdir -p "$BUNDLE_DIR"

echo "[1/5] Build Python dist (wheel/sdist)"
python3 -m build
cp -v dist/* "$BUNDLE_DIR/" || true

echo "[2/5] Copy deployment assets"
cp -v compose.yaml compose.prod.yaml .env.example "$BUNDLE_DIR/"
mkdir -p "$BUNDLE_DIR/docs" "$BUNDLE_DIR/deploy"
cp -v docs/DEPLOYMENT.md docs/DELIVERY.md "$BUNDLE_DIR/docs/"
cp -v README.md LICENSE LICENSING.md COMMERCIAL_LICENSE.md SECURITY.md CHANGELOG.md CITATION.cff "$BUNDLE_DIR/" || true
if [ -d deploy ]; then
  cp -R deploy "$BUNDLE_DIR/"
fi

SKIP_DOCKER="${ZOCR_DELIVERY_SKIP_DOCKER:-0}"
if [ "$SKIP_DOCKER" != "1" ]; then
  echo "[3/5] Build Docker image (api)"
  IMAGE_TAG="${ZOCR_API_IMAGE_TAG:-zocr-suite:${VERSION}}"
  if ! docker info >/dev/null 2>&1; then
    cat <<EOF >&2
[ERROR] Docker daemon is not available.
- Start Docker Desktop (or the docker daemon) and re-run, OR
- set ZOCR_DELIVERY_SKIP_DOCKER=1 to build a Python-only bundle.
EOF
    exit 3
  fi
  docker build -t "$IMAGE_TAG" --build-arg ZOCR_EXTRAS="api" .

  echo "[4/5] Save Docker image"
  docker save "$IMAGE_TAG" -o "$BUNDLE_DIR/zocr-suite-${VERSION}-docker.tar"
else
  echo "[3/5] Skip Docker image build/save (ZOCR_DELIVERY_SKIP_DOCKER=1)"
  IMAGE_TAG=""
fi

echo "[5/5] Generate SHA256SUMS"
(
  cd "$BUNDLE_DIR"
  python3 -S - <<'PY' > SHA256SUMS
import hashlib
from pathlib import Path

root = Path(".")
paths = [p for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS"]
for path in sorted(paths):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    print(f"{h.hexdigest()}  {path.as_posix()}")
PY
)

cat <<EOF

Done.
- Bundle: $BUNDLE_DIR
- Docker tag: ${IMAGE_TAG:-"(skipped)"}

Next:
  bash scripts/verify_delivery.sh "$BUNDLE_DIR/SHA256SUMS"
EOF

if [ "$SKIP_DOCKER" != "1" ]; then
  cat <<EOF

Docker:
  docker load -i "$BUNDLE_DIR/zocr-suite-${VERSION}-docker.tar"
  export ZOCR_API_IMAGE="$IMAGE_TAG"
  docker compose up -d --no-build
EOF
fi
