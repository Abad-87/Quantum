"""
hiring/predictor.py

Prediction logic for the Job Hiring domain.

What this file does:
1. Takes validated input features (no sensitive attributes).
2. Passes them to the hiring model.
3. Generates a human-readable explanation using feature importance.
4. Returns prediction (hired / not hired) + explanation.
"""

import numpy as np
import logging
import pandas as pd

logger = logging.getLogger("hiring.predictor")

# ─── Feature names the model expects ────────────────────────────────────────
# These must match the columns used when training hiring_model.pkl
FEATURE_NAMES = [
    "years_experience",
    "education_level",       # Encoded: 0=High School, 1=Bachelor, 2=Master, 3=PhD
    "technical_score",       # 0–100
    "communication_score",   # 0–100
    "num_past_jobs",
    "certifications",        # Count of relevant certifications
]

# Human-readable labels for education_level encoding
EDUCATION_MAP = {0: "High School", 1: "Bachelor's", 2: "Master's", 3: "PhD"}


def predict(model, features: dict) -> tuple[int, float, str]:
    """
    Run hiring prediction.

    Args:
        model:    Loaded sklearn model/pipeline.
        features: Dict of input features (already validated, no sensitive attrs).

    Returns:
        (prediction, confidence, explanation)
        prediction:  1 = Hired, 0 = Not Hired
        confidence:  probability score (0.0 – 1.0)
        explanation: Human-readable reason string
    """
    # ── Build input array in the exact column order the model expects ────────
    input_row = [[
        features["years_experience"],
        features["education_level"],
        features["technical_score"],
        features["communication_score"],
        features["num_past_jobs"],
        features["certifications"],
    ]]

    # ── Model prediction ─────────────────────────────────────────────────────
    prediction = int(model.predict(input_row)[0])

    # ── Confidence score (probability of positive class) ─────────────────────
    confidence = 0.5  # Default if model doesn't support predict_proba
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_row)[0]
        confidence = round(float(proba[1]), 3)

    # ── Explanation (feature importance-based) ────────────────────────────────
    explanation = _explain(model, features, prediction, input_row)

    return prediction, confidence, explanation


def _explain(model, features: dict, prediction: int, input_row: list) -> str:
    """
    Generates a plain-English explanation.

    Strategy:
    1. Try to get SHAP values if shap is installed.
    2. Fall back to rule-based explanation using feature values.

    This is intentionally simple — perfect for a hackathon demo.
    """
    # ── Try SHAP first ────────────────────────────────────────────────────────
    try:
        import shap
        base_model = model  # If it's a Pipeline, grab the final step
        if hasattr(model, "steps"):
            base_model = model.steps[-1][1]

        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(input_row)

        # For binary classification, shap_values may be a list [neg_class, pos_class]
        if isinstance(shap_values, list):
            values = shap_values[prediction]
        else:
            values = shap_values[0]

        # Map SHAP values back to feature names
        shap_pairs = sorted(
            zip(FEATURE_NAMES, values[0] if hasattr(values[0], "__iter__") else values),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        top_feature, top_value = shap_pairs[0]
        direction = "high" if top_value > 0 else "low"
        pretty_name = top_feature.replace("_", " ")

        if prediction == 1:
            return f"Hired — primarily driven by {direction} {pretty_name} (score: {features.get(top_feature, 'N/A')})."
        else:
            return f"Not hired — primarily due to {direction} {pretty_name} (score: {features.get(top_feature, 'N/A')})."

    except Exception as e:
        logger.debug(f"SHAP not available or failed ({e}). Using rule-based explanation.")

    # ── Rule-based fallback explanation ──────────────────────────────────────
    return _rule_based_explanation(features, prediction)


def _rule_based_explanation(features: dict, prediction: int) -> str:
    """
    Simple rule-based explanation — readable, no external libraries needed.
    Identifies the weakest area and explains the decision.
    """
    tech   = features.get("technical_score", 0)
    comm   = features.get("communication_score", 0)
    exp    = features.get("years_experience", 0)
    edu    = features.get("education_level", 0)
    certs  = features.get("certifications", 0)

    if prediction == 1:  # Hired
        strengths = []
        if tech >= 70:
            strengths.append(f"strong technical score ({tech}/100)")
        if comm >= 70:
            strengths.append(f"strong communication score ({comm}/100)")
        if exp >= 3:
            strengths.append(f"{exp} years of experience")
        if certs > 0:
            strengths.append(f"{certs} certification(s)")

        reason = ", ".join(strengths) if strengths else "overall solid profile"
        return f"Hired — candidate shows {reason}."

    else:  # Not Hired
        weaknesses = []
        if tech < 50:
            weaknesses.append(f"low technical score ({tech}/100)")
        if comm < 50:
            weaknesses.append(f"low communication score ({comm}/100)")
        if exp < 2:
            weaknesses.append(f"limited experience ({exp} years)")
        if edu == 0:
            weaknesses.append("no degree beyond high school")

        reason = ", ".join(weaknesses) if weaknesses else "did not meet minimum requirements"
        return f"Not hired — {reason}."
