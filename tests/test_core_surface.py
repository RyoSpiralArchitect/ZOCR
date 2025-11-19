import importlib
import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def reload_core_surface():
    """Reset the lazy surface cache for isolated assertions."""

    surface = importlib.import_module("zocr.core._surface")
    surface._LOADED.clear()
    importlib.reload(surface)

    core_pkg = importlib.import_module("zocr.core")
    importlib.reload(core_pkg)
    try:
        yield
    finally:
        surface._LOADED.clear()


def test_surface_exports_proxy_functions():
    import zocr.core as core
    import zocr.core.zocr_core as module

    for name in module.__all__:
        exported = getattr(core, name)
        assert exported is getattr(module, name), name


def test_surface_exposes_module_directly():
    import zocr.core as core
    import zocr.core.zocr_core as module

    assert core.zocr_core is module


def test_surface_dir_matches_exports():
    import zocr.core as core

    exported = set(core.__all__)
    assert exported <= set(dir(core))


def test_unknown_attribute_raises():
    import zocr.core as core

    with pytest.raises(AttributeError):
        _ = core.does_not_exist
