"""
Z-OCR Modules / Z-OCR モジュール群 / Modules Z-OCR

- [JA] `zocr.consensus.zocr_consensus` は単一ファイル OCR+テーブル復元。
- [EN] `zocr.core.zocr_core` provides the multi-domain augmentation/index/monitor core.
- [FR] `zocr.orchestrator.zocr_pipeline` orchestre la chaîne de bout en bout.
"""

from importlib import import_module as _import_module
import sys as _sys

zocr_consensus = _import_module(".consensus.zocr_consensus", __name__)
zocr_core = _import_module(".core.zocr_core", __name__)
zocr_pipeline = _import_module(".orchestrator.zocr_pipeline", __name__)

__all__ = [
    "zocr_consensus",
    "zocr_core",
    "zocr_pipeline",
]

_sys.modules.setdefault("zocr_onefile_consensus", zocr_consensus)
_sys.modules.setdefault("zocr_multidomain_core", zocr_core)
_sys.modules.setdefault("zocr_pipeline_allinone", zocr_pipeline)
