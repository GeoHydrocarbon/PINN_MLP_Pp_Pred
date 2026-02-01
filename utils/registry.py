import json
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_REGISTRY_PATH = Path("models") / "registry.json"


def load_registry(registry_path: Optional[Path] = None) -> Dict[str, Any]:
    path = registry_path or DEFAULT_REGISTRY_PATH
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_registry(
    registry: Dict[str, Any],
    registry_path: Optional[Path] = None,
) -> Path:
    path = registry_path or DEFAULT_REGISTRY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
    return path


def upsert_registry(
    alias: str,
    entry: Dict[str, Any],
    registry_path: Optional[Path] = None,
) -> Dict[str, Any]:
    registry = load_registry(registry_path)
    registry[alias] = entry
    save_registry(registry, registry_path)
    return entry


def get_registry_entry(alias: str, registry_path: Optional[Path] = None) -> Dict[str, Any]:
    registry = load_registry(registry_path)
    if alias not in registry:
        raise KeyError(f"Alias {alias} not found in registry.")
    return registry[alias]
