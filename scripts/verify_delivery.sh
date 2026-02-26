#!/usr/bin/env bash
set -euo pipefail

SUMS_FILE="${1:-}"
if [ -z "$SUMS_FILE" ]; then
  echo "Usage: $0 <path-to-SHA256SUMS>" >&2
  exit 2
fi

DIR="$(cd "$(dirname "$SUMS_FILE")" && pwd)"
cd "$DIR"

python3 -S - <<'PY'
import hashlib
from pathlib import Path

sums = Path("SHA256SUMS").read_text(encoding="utf-8").splitlines()
ok = True
for line in sums:
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    try:
        digest, rel = line.split(None, 1)
        rel = rel.strip()
    except ValueError:
        print("[ERROR] invalid line:", line)
        ok = False
        continue
    path = Path(rel)
    if not path.exists() or not path.is_file():
        print("[ERROR] missing:", rel)
        ok = False
        continue
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != digest:
        print("[ERROR] mismatch:", rel)
        print("  expected:", digest)
        print("  actual:  ", actual)
        ok = False
    else:
        print("[OK] ", rel)
raise SystemExit(0 if ok else 1)
PY
