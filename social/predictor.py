"""
social/predictor.py  —  Phase 3: async SHAP

Critical-path changes: SHAP removed.  See hiring/predictor.py for full notes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

from fairness.checker import compute_bias_risk_score

logger = logging.getLogger("social.predictor")

FEATURE_NAMES = [
    "avg_session_minutes",
    "posts_per_day",
    "topics_interacted",
    "like_rate",
    "share_rate",
    "comment_rate",
    "account_age_days",
]

CONTENT_CATEGORIES: Dict[int, str] = {
    0: "Technology & Science",
    1: "Entertainment & Pop Culture",
    2: "Sports & Fitness",
    3: "News & Politics",
    4: "Arts & Creativity",
    5: "Education & Learning",
    6: "Business & Finance",
    7: "Lifestyle & Wellness",
}


def predict(
    model,
    features:       Dict[str, Any],
    sensitive_attr: Optional[str] = None,
    domain:         str = "social",
) -> dict:
    """Fast synchronous prediction — SHAP computed asynchronously."""
    input_row      = _build_input_row(features)
    category_id    = int(model.predict(input_row)[0])
    category_label = CONTENT_CATEGORIES.get(category_id, f"Category {category_id}")

    confidence = 0.5
    if hasattr(model, "predict_proba"):
        proba      = model.predict_proba(input_row)[0]
        confidence = round(float(np.max(proba)), 4)

    bias_risk = compute_bias_risk_score(
        confidence     = confidence,
        shap_values    = None,
        sensitive_attr = sensitive_attr,
        domain         = domain,
    )

    explanation = _rule_based_explanation(features, category_label)

    return {
        "prediction":     category_id,
        "category_label": category_label,
        "confidence":     confidence,
        "shap_values":    {},
        "shap_available": False,
        "shap_status":    "pending",
        "explanation":    explanation,
        "bias_risk":      bias_risk,
        "input_row":      input_row,
        "feature_names":  FEATURE_NAMES,
    }


def _build_input_row(features: dict) -> list:
    return [[
        features["avg_session_minutes"],
        features["posts_per_day"],
        features["topics_interacted"],
        features["like_rate"],
        features["share_rate"],
        features["comment_rate"],
        features["account_age_days"],
    ]]


def _rule_based_explanation(features: dict, category_label: str) -> str:
    like_rate    = features.get("like_rate", 0)
    share_rate   = features.get("share_rate", 0)
    session_mins = features.get("avg_session_minutes", 0)
    posts        = features.get("posts_per_day", 0)

    signals = []
    if like_rate    > 0.6: signals.append("high content engagement")
    if share_rate   > 0.3: signals.append("active content sharing")
    if session_mins > 30:  signals.append(f"long sessions ({session_mins:.0f} min avg)")
    if posts        > 2:   signals.append("frequent posting")

    signals_str = ", ".join(signals) or "your recent activity patterns"
    return (
        f"Recommended '{category_label}' based on {signals_str}. "
        "(Full SHAP explanation pending)"
    )
