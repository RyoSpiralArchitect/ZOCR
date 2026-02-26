#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VERSION="$(python3 -S -c 'from zocr._version import __version__; print(__version__)')"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

OUT_ROOT="${ZOCR_DELIVERY_DIR:-delivery}"
BUNDLE_DIR="${OUT_ROOT}/zocr-suite-${VERSION}-${STAMP}"

mkdir -p "$BUNDLE_DIR"

echo "[1/7] Build Python dist (wheel/sdist)"
python3 -m build
cp -v dist/* "$BUNDLE_DIR/" || true

echo "[2/7] Copy deployment assets"
cp -v compose.yaml compose.prod.yaml .env.example "$BUNDLE_DIR/"
mkdir -p "$BUNDLE_DIR/docs" "$BUNDLE_DIR/deploy"
cp -v docs/DEPLOYMENT.md docs/DELIVERY.md docs/COMPLIANCE.md "$BUNDLE_DIR/docs/" 2>/dev/null || true
cp -v README.md LICENSE LICENSING.md COMMERCIAL_LICENSE.md SECURITY.md CHANGELOG.md CITATION.cff "$BUNDLE_DIR/" || true
if [ -d deploy ]; then
  cp -R deploy "$BUNDLE_DIR/"
fi

DELIVERY_COMPLIANCE="${ZOCR_DELIVERY_COMPLIANCE:-0}"
if [ "$DELIVERY_COMPLIANCE" = "1" ]; then
  echo "[3/7] Generate compliance artifacts (optional)"
  ZOCR_COMPLIANCE_OUTDIR="$BUNDLE_DIR/compliance" \
    ZOCR_COMPLIANCE_EXTRAS="${ZOCR_DELIVERY_COMPLIANCE_EXTRAS:-api}" \
    bash scripts/generate_compliance_artifacts.sh
else
  echo "[3/7] Skip compliance artifacts (ZOCR_DELIVERY_COMPLIANCE=0)"
fi

DELIVERY_WHEELS="${ZOCR_DELIVERY_WHEELS:-0}"
if [ "$DELIVERY_WHEELS" = "1" ]; then
  echo "[4/7] Download wheels for offline installs (optional)"
  WHEELS_DIR="$BUNDLE_DIR/wheels"
  mkdir -p "$WHEELS_DIR"
  TMP_VENV="$(mktemp -d)"
  python3 -m venv "$TMP_VENV/venv"
  # shellcheck disable=SC1091
  source "$TMP_VENV/venv/bin/activate"
  python -m pip install --upgrade pip >/dev/null
  EXTRAS="$(echo "${ZOCR_DELIVERY_WHEEL_EXTRAS:-api}" | tr -d '[:space:]')"
  SPEC="."
  if [ -n "$EXTRAS" ]; then
    SPEC=".[${EXTRAS}]"
  fi
  python -m pip download --dest "$WHEELS_DIR" "$SPEC"
  deactivate
  rm -rf "$TMP_VENV"
else
  echo "[4/7] Skip offline wheels (ZOCR_DELIVERY_WHEELS=0)"
fi

SKIP_DOCKER="${ZOCR_DELIVERY_SKIP_DOCKER:-0}"
if [ "$SKIP_DOCKER" != "1" ]; then
  echo "[5/7] Build Docker image (api)"
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

  echo "[6/7] Save Docker image"
  docker save "$IMAGE_TAG" -o "$BUNDLE_DIR/zocr-suite-${VERSION}-docker.tar"
else
  echo "[5/7] Skip Docker image build/save (ZOCR_DELIVERY_SKIP_DOCKER=1)"
  IMAGE_TAG=""
fi

echo "[7/7] Generate SHA256SUMS"
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
