"""Path resolution for RiTAL.

Configs refer to data and output locations through two variables so that the
repository contains no machine-specific absolute paths:

``${RITAL_DATA}``
    Where the training data lives. Defaults to ``<repo>/data``.
``${RITAL_RUNS}``
    Where training writes checkpoints and logs. Defaults to ``<repo>/runs``.

Override either by exporting the environment variable of the same name::

    export RITAL_DATA=/scratch/me/rital-data
    export RITAL_RUNS=/scratch/me/rital-runs

``expand_config`` is applied to a config right after it is parsed, so every
path-valued field in the YAML goes through the same substitution.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def data_root() -> Path:
    return Path(os.environ.get("RITAL_DATA") or (REPO_ROOT / "data")).expanduser()


def runs_root() -> Path:
    return Path(os.environ.get("RITAL_RUNS") or (REPO_ROOT / "runs")).expanduser()


def _mapping() -> dict[str, str]:
    env = dict(os.environ)
    env["RITAL_DATA"] = str(data_root())
    env["RITAL_RUNS"] = str(runs_root())
    return env


def expand(value: str) -> str:
    """Expand ``${RITAL_DATA}``/``${RITAL_RUNS}``/other env vars and ``~``."""
    env = _mapping()
    out = os.path.expandvars(value) if "$" not in value else _expand_with(value, env)
    return os.path.expanduser(out)


def _expand_with(value: str, env: dict[str, str]) -> str:
    saved = os.environ.copy()
    try:
        os.environ.update(env)
        return os.path.expandvars(value)
    finally:
        os.environ.clear()
        os.environ.update(saved)


def expand_config(obj):
    """Recursively expand path variables in every string inside a parsed config."""
    if isinstance(obj, str):
        return expand(obj) if "$" in obj or obj.startswith("~") else obj
    if isinstance(obj, dict):
        return {k: expand_config(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_config(v) for v in obj]
    return obj
