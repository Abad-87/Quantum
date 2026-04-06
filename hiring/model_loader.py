"""
hiring/model_loader.py

Thin adapter that delegates all model-lifecycle work to the shared
ModelRegistry singleton.  Business code should always call get_model()
or get_model_ab() instead of touching joblib directly.

Hot-swap without a server restart::

    from hiring.model_loader import hot_swap
    from pathlib import Path
    hot_swap(Path("models/hiring_model_v2.pkl"))

A/B fairness test (10 % of traffic to a challenger)::

    from utils.model_registry import registry
    from pathlib import Path
    registry.register_ab_variant(
        "hiring", "challenger",
        Path("models/hiring_model_v2.pkl"),
        traffic_fraction=0.10,
    )
"""

from pathlib import Path
from typing import Any, Tuple

from utils.model_registry import registry

MODEL_NAME = "hiring"
MODEL_PATH = Path("models/hiring_model.pkl")


def preload(path: Path = MODEL_PATH) -> None:
    """
    Load the hiring model into memory.
    Called once at application startup via main.py lifespan hook.
    """
    registry.load(MODEL_NAME, path)


def get_model(variant: str = "primary") -> Any:
    """
    Return the cached hiring model for *variant* (default: "primary").
    Raises KeyError if preload() has not been called yet.
    """
    return registry.get(MODEL_NAME, variant)


def get_model_ab() -> Tuple[Any, str]:
    """
    Return *(model, variant_name)* respecting any active A/B traffic split.
    Falls back to ("primary model", "primary") when no split is configured.
    """
    return registry.get_ab(MODEL_NAME)


def get_version(variant: str = "primary") -> str:
    """12-character version hex for the given variant (for audit logging)."""
    return registry.get_version(MODEL_NAME, variant)


def get_metadata(variant: str = "primary") -> dict:
    """Full provenance dict — embedded in the correlation audit log."""
    return registry.get_metadata(MODEL_NAME, variant)


def hot_swap(new_path: Path = MODEL_PATH, variant: str = "primary"):
    """
    Atomically reload the hiring model from *new_path* at runtime.
    No server restart required.  Safe to call while the server is handling
    live traffic — in-flight requests finish on the old model.
    """
    return registry.hot_swap(MODEL_NAME, new_path, variant)
