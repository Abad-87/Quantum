"""
hiring/model_loader.py

Loads the hiring_model.pkl file exactly once when the app starts.
Uses joblib for fast deserialization of scikit-learn models.

Why load once?  Loading a .pkl file on every request is slow.
We cache it in memory using a module-level variable.
"""

import os
import joblib
import logging
from pathlib import Path

logger = logging.getLogger("hiring.model_loader")

# Path to the model file (relative to project root)
MODEL_PATH = Path("models/hiring_model.pkl")

# Module-level cache — loaded once, reused on every request
_model = None


def load_model():
    """
    Loads the hiring model from disk into memory.
    Returns the model object (e.g. sklearn Pipeline or Classifier).
    Raises FileNotFoundError if the .pkl file is missing.
    """
    global _model

    if _model is not None:
        return _model  # Already loaded — return cached version

    if not MODEL_PATH.exists():
        logger.error(f"Model file not found: {MODEL_PATH}")
        raise FileNotFoundError(
            f"hiring_model.pkl not found at '{MODEL_PATH}'. "
            "Please place the model file in the /models directory."
        )

    logger.info(f"Loading hiring model from {MODEL_PATH} …")
    _model = joblib.load(MODEL_PATH)
    logger.info("Hiring model loaded successfully.")
    return _model


def get_model():
    """Convenience function — returns cached model or loads it."""
    return load_model()
