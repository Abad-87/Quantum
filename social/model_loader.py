"""
social/model_loader.py
Loads social_model.pkl once at startup.
"""

import logging
from pathlib import Path
import joblib

logger = logging.getLogger("social.model_loader")

MODEL_PATH = Path("models/social_model.pkl")
_model = None


def get_model():
    global _model
    if _model is not None:
        return _model

    if not MODEL_PATH.exists():
        logger.error(f"Model file not found: {MODEL_PATH}")
        raise FileNotFoundError(
            f"social_model.pkl not found at '{MODEL_PATH}'. "
            "Please place the model file in the /models directory."
        )

    logger.info(f"Loading social recommendation model from {MODEL_PATH} …")
    _model = joblib.load(MODEL_PATH)
    logger.info("Social model loaded successfully.")
    return _model
