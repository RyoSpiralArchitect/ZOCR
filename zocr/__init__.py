"""
Z-OCR Modules / Z-OCR モジュール群 / Modules Z-OCR

- [JA] `zocr.consensus.zocr_consensus` は単一ファイル OCR+テーブル復元。
- [EN] `zocr.core.zocr_core` provides the multi-domain augmentation/index/monitor core.
- [FR] `zocr.orchestrator.zocr_pipeline` orchestre la chaîne de bout en bout.
"""

from importlib import import_module as _import_module
import sys as _sys
from typing import Any

zocr_consensus = _import_module(".consensus.zocr_consensus", __name__)
zocr_core = _import_module(".core.zocr_core", __name__)
_zocr_pipeline = None

__all__ = [
    "zocr_consensus",
    "zocr_core",
    "zocr_pipeline",
]

_sys.modules.setdefault("zocr_onefile_consensus", zocr_consensus)
_sys.modules.setdefault("zocr_multidomain_core", zocr_core)


def _load_pipeline():
    global _zocr_pipeline
    if _zocr_pipeline is None:
        module = _import_module(".orchestrator.zocr_pipeline", __name__)
        _sys.modules.setdefault("zocr_pipeline_allinone", module)
        _zocr_pipeline = module
    return _zocr_pipeline


def __getattr__(name: str) -> Any:
    if name == "zocr_pipeline":
        return _load_pipeline()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__ + list(globals().keys())))
