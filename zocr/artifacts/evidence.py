from __future__ import annotations

import hashlib
import io
from pathlib import Path


def archive_page_image(image, outdir: str | Path) -> str | None:
    if image.width * image.height > 40_000_000:
        return None
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    data = buffer.getvalue()
    if len(data) > 32 * 1024 * 1024:
        return None
    target = Path(outdir) / "evidence" / (hashlib.sha256(data).hexdigest() + ".png")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_bytes(data)
    return str(target.resolve())
