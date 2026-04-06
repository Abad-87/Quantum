"""
loan/model_loader.py

Thin adapter over the shared ModelRegistry singleton for the loan domain.
See hiring/model_loader.py for full usage documentation.
"""

from pathlib import Path
from typing import Any, Tuple

from utils.model_registry import registry

MODEL_NAME = "loan"
MODEL_PATH = Path("models/loan_model.pkl")


def preload(path: Path = MODEL_PATH) -> None:
    """Load the loan model into memory at application startup."""
    registry.load(MODEL_NAME, path)


def get_model(variant: str = "primary") -> Any:
    """Return the cached loan model for *variant*."""
    return registry.get(MODEL_NAME, variant)


def get_model_ab() -> Tuple[Any, str]:
    """Return *(model, variant_name)* respecting any active A/B split."""
    return registry.get_ab(MODEL_NAME)


def get_version(variant: str = "primary") -> str:
    """12-character version hex for audit logging."""
    return registry.get_version(MODEL_NAME, variant)


def get_metadata(variant: str = "primary") -> dict:
    """Full provenance dict for correlation audit embedding."""
    return registry.get_metadata(MODEL_NAME, variant)


def hot_swap(new_path: Path = MODEL_PATH, variant: str = "primary"):
    """
    Atomically reload the loan model from *new_path* at runtime.
    No server restart required.
    """
    return registry.hot_swap(MODEL_NAME, new_path, variant)
