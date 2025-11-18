"""Simple plugin registry for the pipeline orchestrator."""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List

StageHook = Callable[..., None]

# ``PLUGINS`` intentionally remains a plain dict so callers (like the CLI and
# stack inspector) can introspect the registry without going through helper
# functions.
PLUGINS: Dict[str, List[StageHook]] = {}


def register_stage(stage: str) -> Callable[[StageHook], StageHook]:
    """Register ``fn`` under ``stage`` and return the original function."""

    def decorator(fn: StageHook) -> StageHook:
        PLUGINS.setdefault(stage, []).append(fn)
        return fn

    return decorator


def call_stage(stage: str, **kwargs) -> None:
    """Invoke every hook registered for ``stage`` with ``kwargs``."""

    for fn in PLUGINS.get(stage, ()):  # pragma: no branch - trivial loop
        try:
            fn(**kwargs)
        except Exception as exc:  # pragma: no cover - diagnostics only
            fn_name = getattr(fn, "__name__", repr(fn))
            print(f"[PLUGIN:{stage}] {fn_name} -> {exc}")


def describe_plugins() -> Dict[str, List[str]]:
    """Return a serializable view of the registry for summaries/CLI output."""

    return {
        stage: [getattr(fn, "__name__", repr(fn)) for fn in fns]
        for stage, fns in PLUGINS.items()
    }


def iter_stage_functions(stage: str) -> Iterable[StageHook]:
    """Yield the functions registered for ``stage`` (if any)."""

    return tuple(PLUGINS.get(stage, ()))


def clear_plugins() -> None:
    """Reset the registry (useful for tests)."""

    PLUGINS.clear()


__all__ = [
    "PLUGINS",
    "call_stage",
    "clear_plugins",
    "describe_plugins",
    "iter_stage_functions",
    "register_stage",
]
