"""Load YAML configuration and resolve project paths.

Configuration is separated from logic so countries, FX rates, column aliases,
and temporal splits can change without editing Python modules.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def _resolve_config_path(path: Path | str | None) -> Path:
    if path is not None:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate

    nested = PROJECT_ROOT / "configs" / "config.yaml"
    if nested.exists():
        return nested
    return PROJECT_ROOT / "config.yaml"


def _maybe_follow_extends(raw: dict[str, Any], loaded_from: Path) -> dict[str, Any]:
    extends = raw.get("extends")
    if not extends:
        return raw
    target = Path(str(extends))
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    if not target.exists():
        raise FileNotFoundError(f"Config extends missing file: {target} (from {loaded_from})")
    with target.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@lru_cache(maxsize=4)
def load_config(path: str | None = None) -> dict[str, Any]:
    """Load the YAML config. Pass an absolute or repo-relative path to override."""
    config_path = _resolve_config_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return _maybe_follow_extends(raw, config_path)


def resolve_path(relative: str | Path) -> Path:
    """Turn a config-relative path into an absolute path under the repo root."""
    candidate = Path(relative)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def get_path(config: dict[str, Any], key: str) -> Path:
    """Read `paths.<key>` from config and resolve it."""
    relative = config["paths"][key]
    return resolve_path(relative)
