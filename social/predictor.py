"""
social/predictor.py

Prediction logic for the Social Media Recommendation domain.

predict() returns a structured dict:
{
    "prediction":           int,    # category ID (0–7)
    "category_label":       str,    # human-readable category name
    "confidence":           float,
    "shap_values":          dict,   # {feature_name: float}
    "shap_available":       bool,
    "explanation":          str,
    "bias_risk":            dict,   # from compute_bias_risk_score()
}
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


# ─── Main entry point ─────────────────────────────────────────────────────────

def predict(
    model,
    features: Dict[str, Any],
    sensitive_attr: Optional[str] = None,
    domain: str = "social",
) -> dict:
    """
    Run content-category recommendation and return a fully structured result dict.

    Parameters
    ----------
    model          : Loaded sklearn model/pipeline from the registry.
    features       : Validated behavioural features — no sensitive attributes.
    sensitive_attr : Sensitive attribute name for bias-risk weighting only.
    domain         : Domain label forwarded to bias-risk computation.
    """
    input_row = _build_input_row(features)

    # ── Prediction ────────────────────────────────────────────────────────────
    category_id    = int(model.predict(input_row)[0])
    category_label = CONTENT_CATEGORIES.get(category_id, f"Category {category_id}")

    # ── Confidence (max class probability) ───────────────────────────────────
    confidence = 0.5
    if hasattr(model, "predict_proba"):
        proba      = model.predict_proba(input_row)[0]
        confidence = round(float(np.max(proba)), 4)

    # ── SHAP values ───────────────────────────────────────────────────────────
    shap_values, shap_available = _compute_shap(model, input_row, category_id)

    # ── Bias risk ─────────────────────────────────────────────────────────────
    bias_risk = compute_bias_risk_score(
        confidence=confidence,
        shap_values=shap_values,
        sensitive_attr=sensitive_attr,
        domain=domain,
    )

    # ── Explanation ───────────────────────────────────────────────────────────
    explanation = _explain(
        features, category_id, category_label, shap_values, shap_available
    )

    return {
        "prediction":     category_id,
        "category_label": category_label,
        "confidence":     confidence,
        "shap_values":    shap_values,
        "shap_available": shap_available,
        "explanation":    explanation,
        "bias_risk":      bias_risk,
    }


# ─── Internals ────────────────────────────────────────────────────────────────

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


def _compute_shap(
    model,
    input_row: list,
    category_id: int,
) -> tuple[Dict[str, float], bool]:
    """
    For multiclass models, extract SHAP values for the predicted class only,
    so the returned dict is always {feature: scalar} — no nested arrays.
    """
    try:
        import shap

        base_model = model.steps[-1][1] if hasattr(model, "steps") else model
        explainer  = shap.TreeExplainer(base_model)
        raw        = explainer.shap_values(input_row)

        # Multiclass: raw is List[n_classes × n_samples × n_features]
        if isinstance(raw, list) and len(raw) > category_id:
            flat = raw[category_id][0]
        elif isinstance(raw, np.ndarray):
            flat = raw[0]
        else:
            raise ValueError(f"Unexpected SHAP output type: {type(raw)}")

        shap_dict = {
            feat: round(float(val), 6)
            for feat, val in zip(FEATURE_NAMES, flat)
        }
        return shap_dict, True

    except Exception as exc:
        logger.debug(f"SHAP unavailable: {exc}")
        return {}, False


def _explain(
    features: dict,
    category_id: int,
    category_label: str,
    shap_values: Dict[str, float],
    shap_available: bool,
) -> str:
    if shap_available and shap_values:
        return _shap_explanation(features, category_label, shap_values)
    return _rule_based_explanation(features, category_label)


def _shap_explanation(
    features: dict,
    category_label: str,
    shap_values: Dict[str, float],
) -> str:
    top_feat, _ = max(shap_values.items(), key=lambda kv: abs(kv[1]))
    pretty      = top_feat.replace("_", " ")
    feat_value  = features.get(top_feat, "N/A")
    return (
        f"Recommended '{category_label}' based on your engagement patterns, "
        f"primarily driven by {pretty} ({feat_value})."
    )


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
        "This recommendation is based on behaviour only, not personal attributes."
    )
