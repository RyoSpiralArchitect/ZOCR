#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUTDIR="${ZOCR_COMPLIANCE_OUTDIR:-compliance}"
EXTRAS_RAW="${ZOCR_COMPLIANCE_EXTRAS:-api}"
PYTHON="${ZOCR_COMPLIANCE_PYTHON:-python3}"
KEEP_TEMP="${ZOCR_COMPLIANCE_KEEP_TEMP:-0}"
GATHER_LICENSE_TEXTS="${ZOCR_COMPLIANCE_GATHER_LICENSE_TEXTS:-0}"

EXTRAS="$(echo "$EXTRAS_RAW" | tr -d '[:space:]')"

RUNTIME_DIR="$(mktemp -d)"
TOOL_DIR="$(mktemp -d)"
cleanup() {
  if [ "$KEEP_TEMP" = "1" ]; then
    echo "[INFO] Keeping temp dirs:"
    echo "  runtime: $RUNTIME_DIR"
    echo "  tool:    $TOOL_DIR"
    return
  fi
  rm -rf "$RUNTIME_DIR" "$TOOL_DIR"
}
trap cleanup EXIT

mkdir -p "$OUTDIR"

echo "[1/4] Create runtime venv"
"$PYTHON" -m venv "$RUNTIME_DIR/venv"
source "$RUNTIME_DIR/venv/bin/activate"
python -m pip install --upgrade pip >/dev/null

echo "[2/4] Install Z-OCR + deps"
if [ -n "$EXTRAS" ]; then
  python -m pip install ".[${EXTRAS}]"
else
  python -m pip install "."
fi

python -m pip freeze --all > "$OUTDIR/requirements.lock.txt"

echo "[3/4] Generate third-party notices"
python scripts/generate_third_party_notices.py -o "$OUTDIR/THIRD_PARTY_NOTICES.md"

deactivate

echo "[4/4] Generate Python SBOM (CycloneDX)"
"$PYTHON" -m venv "$TOOL_DIR/venv"
source "$TOOL_DIR/venv/bin/activate"
python -m pip install --upgrade pip >/dev/null
python -m pip install cyclonedx-bom >/dev/null

SBOM_ARGS=(environment --output-reproducible --pyproject pyproject.toml -o "$OUTDIR/sbom-python.cdx.json")
if [ "$GATHER_LICENSE_TEXTS" = "1" ]; then
  SBOM_ARGS+=(--gather-license-texts)
fi
SBOM_ARGS+=("$RUNTIME_DIR/venv/bin/python")

cyclonedx-py "${SBOM_ARGS[@]}"

echo
echo "Done."
echo "- Outdir: $OUTDIR"
echo "- Files:"
echo "  - $OUTDIR/THIRD_PARTY_NOTICES.md"
echo "  - $OUTDIR/sbom-python.cdx.json"
echo "  - $OUTDIR/requirements.lock.txt"
