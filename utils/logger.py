"""
utils/logger.py

Centralized logging setup.
Every prediction is logged with:
  - domain
  - input features (no sensitive data)
  - prediction result
  - fairness metrics
  - timestamp
"""

import logging
import json
import os
from datetime import datetime, timezone


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Creates a logger with consistent formatting.
    Usage:  logger = setup_logger("hiring")
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    logger.setLevel(level)
    return logger


# Module-level prediction logger
_pred_logger = setup_logger("predictions")


def log_prediction(
    domain: str,
    input_data: dict,
    prediction: int | float,
    prediction_label: str,
    explanation: str,
    fairness_result: dict,
):
    """
    Logs a single prediction event as a structured JSON line.
    This is the audit trail — useful for fairness monitoring over time.

    NOTE: We deliberately EXCLUDE sensitive attributes from the log
    (they are only logged in fairness_result without identifiers).
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "domain": domain,
        "input": input_data,          # Features used for prediction (no sensitive attrs)
        "prediction": prediction,
        "prediction_label": prediction_label,
        "explanation": explanation,
        "fairness": {
            "sensitive_attribute": fairness_result.get("sensitive_attribute"),
            "is_fair": fairness_result.get("is_fair"),
            "warning": fairness_result.get("warning"),
        },
    }

    _pred_logger.info(json.dumps(record))
