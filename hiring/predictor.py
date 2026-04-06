"""
hiring/predictor.py

Prediction logic for the Job Hiring domain.

predict() now returns a fully structured dict instead of a raw tuple so that
routers, loggers, and tests all share the same schema:

{
    "prediction":      int,            # 1 = Hired, 0 = Not Hired
    "confidence":      float,          # probability of positive class (0–1)
    "shap_values":     dict,           # {feature_name: float}  — {} if unavailable
    "shap_available":  bool,
    "explanation":     str,            # human-readable reason
    "bias_risk":       dict,           # output of compute_bias_risk_score()
}

bias_risk is populated here so the router can embed it in the API response
and the correlation logger can record it alongside the prediction.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

from fairness.checker import compute_bias_risk_score

logger = logging.getLogger("hiring.predictor")

# ─── Feature order expected by the model ─────────────────────────────────────
FEATURE_NAMES = [
    "years_experience",
    "education_level",      # 0=High School … 3=PhD
    "technical_score",      # 0–100
    "communication_score",  # 0–100
    "num_past_jobs",
    "certifications",
]

EDUCATION_MAP = {0: "High School", 1: "Bachelor's", 2: "Master's", 3: "PhD"}


# ─── Main entry point ─────────────────────────────────────────────────────────

def predict(
    model,
    features: Dict[str, Any],
    sensitive_attr: Optional[str] = None,
    domain: str = "hiring",
) -> dict:
    """
    Run hiring prediction and return a fully structured result dict.

    Parameters
    ----------
    model          : Loaded sklearn model/pipeline from the registry.
    features       : Validated prediction features — no sensitive attributes.
    sensitive_attr : Sensitive attribute name for bias-risk weighting only.
    domain         : Domain label forwarded to bias-risk computation.

    Returns
    -------
    Structured dict (see module docstring).
    """
    input_row = _build_input_row(features)

    # ── Model prediction ──────────────────────────────────────────────────────
    prediction = int(model.predict(input_row)[0])

    # ── Confidence (probability of positive class) ────────────────────────────
    confidence = 0.5
    if hasattr(model, "predict_proba"):
        proba      = model.predict_proba(input_row)[0]
        confidence = round(float(proba[1]), 4)

    # ── SHAP values ───────────────────────────────────────────────────────────
    shap_values, shap_available = _compute_shap(model, input_row, prediction)

    # ── Bias risk score ───────────────────────────────────────────────────────
    bias_risk = compute_bias_risk_score(
        confidence=confidence,
        shap_values=shap_values,
        sensitive_attr=sensitive_attr,
        domain=domain,
    )

    # ── Human-readable explanation ────────────────────────────────────────────
    explanation = _explain(features, prediction, shap_values, shap_available)

    return {
        "prediction":     prediction,
        "confidence":     confidence,
        "shap_values":    shap_values,
        "shap_available": shap_available,
        "explanation":    explanation,
        "bias_risk":      bias_risk,
    }


# ─── Internals ────────────────────────────────────────────────────────────────

def _build_input_row(features: dict) -> list:
    return [[
        features["years_experience"],
        features["education_level"],
        features["technical_score"],
        features["communication_score"],
        features["num_past_jobs"],
        features["certifications"],
    ]]


def _compute_shap(
    model,
    input_row: list,
    prediction: int,
) -> tuple[Dict[str, float], bool]:
    """
    Attempt SHAP TreeExplainer.  Returns ({feature: value}, True) on success,
    ({}, False) on failure (e.g. shap not installed, unsupported model type).
    """
    try:
        import shap  # optional dependency

        base_model = model.steps[-1][1] if hasattr(model, "steps") else model
        explainer  = shap.TreeExplainer(base_model)
        raw        = explainer.shap_values(input_row)

        # Binary classification: raw is [neg_class_array, pos_class_array]
        values = raw[prediction] if isinstance(raw, list) else raw

        # Flatten to 1-D
        flat = values[0] if hasattr(values[0], "__iter__") else values

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
    prediction: int,
    shap_values: Dict[str, float],
    shap_available: bool,
) -> str:
    if shap_available and shap_values:
        return _shap_explanation(features, prediction, shap_values)
    return _rule_based_explanation(features, prediction)


def _shap_explanation(
    features: dict,
    prediction: int,
    shap_values: Dict[str, float],
) -> str:
    top_feat, top_val = max(shap_values.items(), key=lambda kv: abs(kv[1]))
    direction  = "high" if top_val > 0 else "low"
    pretty     = top_feat.replace("_", " ")
    feat_value = features.get(top_feat, "N/A")

    if prediction == 1:
        return (
            f"Hired — primarily driven by {direction} {pretty} "
            f"(value: {feat_value})."
        )
    return (
        f"Not hired — primarily due to {direction} {pretty} "
        f"(value: {feat_value})."
    )


def _rule_based_explanation(features: dict, prediction: int) -> str:
    tech  = features.get("technical_score", 0)
    comm  = features.get("communication_score", 0)
    exp   = features.get("years_experience", 0)
    certs = features.get("certifications", 0)

    if prediction == 1:
        strengths = []
        if tech  >= 70: strengths.append(f"strong technical score ({tech}/100)")
        if comm  >= 70: strengths.append(f"strong communication score ({comm}/100)")
        if exp   >= 3:  strengths.append(f"{exp} years of experience")
        if certs  > 0:  strengths.append(f"{certs} certification(s)")
        reason = ", ".join(strengths) or "overall solid profile"
        return f"Hired — candidate shows {reason}."

    weaknesses = []
    if tech < 50: weaknesses.append(f"low technical score ({tech}/100)")
    if comm < 50: weaknesses.append(f"low communication score ({comm}/100)")
    if exp  < 2:  weaknesses.append(f"limited experience ({exp} years)")
    if features.get("education_level", 1) == 0:
        weaknesses.append("no degree beyond high school")
    reason = ", ".join(weaknesses) or "did not meet minimum requirements"
    return f"Not hired — {reason}."
