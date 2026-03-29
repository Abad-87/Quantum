"""
loan/model_loader.py
Loads loan_model.pkl once at startup and caches it in memory.
"""

import logging
from pathlib import Path
import joblib

logger = logging.getLogger("loan.model_loader")

MODEL_PATH = Path("models/loan_model.pkl")
_model = None


def get_model():
    global _model
    if _model is not None:
        return _model

    if not MODEL_PATH.exists():
        logger.error(f"Model file not found: {MODEL_PATH}")
        raise FileNotFoundError(
            f"loan_model.pkl not found at '{MODEL_PATH}'. "
            "Please place the model file in the /models directory."
        )

    logger.info(f"Loading loan model from {MODEL_PATH} …")
    _model = joblib.load(MODEL_PATH)
    logger.info("Loan model loaded successfully.")
    return _model
