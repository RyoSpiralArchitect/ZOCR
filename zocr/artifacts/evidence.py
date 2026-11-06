from __future__ import annotations

import hashlib
import io
import os
import tempfile
from pathlib import Path

from .manifest import _sha256_file


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
    if target.is_file() and not target.is_symlink() and _sha256_file(target) == target.stem:
        return str(target.resolve())
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return str(target.resolve())
