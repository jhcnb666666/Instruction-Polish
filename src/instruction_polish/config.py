"""Simple configuration loader."""

import json
from pathlib import Path
from typing import Any

DEFAULT_CONFIG = {
    "default_strategy": "default",
    "output_format": "text",
}


def load_config(path: str | None = None) -> dict[str, Any]:
    """Load JSON config from path or return defaults."""
    if path is None:
        # Look for config in current directory
        candidate = Path("instruction_polish.json")
        if candidate.exists():
            path = str(candidate)
        else:
            return DEFAULT_CONFIG.copy()

    config_path = Path(path)
    if not config_path.exists():
        return DEFAULT_CONFIG.copy()

    with config_path.open(encoding="utf-8") as f:
        user_config = json.load(f)

    config = DEFAULT_CONFIG.copy()
    config.update(user_config)
    return config
