"""
hiring/predictor.py  —  Phase 3: async SHAP

Critical-path changes
---------------------
predict() no longer runs SHAP TreeExplainer.  It performs only:
  1. model.predict()          — integer label
  2. model.predict_proba()    — confidence score
  3. compute_bias_risk_score() — using empty shap_values (Phase-1 weights)
  4. rule-based explanation   — instant, no SHAP dependency

The response is returned to the client immediately.  SHAP is computed in a
BackgroundTask via utils.shap_cache.compute_shap_background(), which runs
the TreeExplainer in FastAPI's ThreadPoolExecutor, stores the result in
ShapCache, and broadcasts it over WebSocket to any subscribed client.

Clients retrieve the full SHAP report by:
  - Polling  GET  /shap/{correlation_id}
  - Listening on  ws://<host>/shap/ws/{correlation_id}
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

from fairness.checker import compute_bias_risk_score

logger = logging.getLogger("hiring.predictor")

FEATURE_NAMES = [
    "years_experience",
    "education_level",
    "technical_score",
    "communication_score",
    "num_past_jobs",
    "certifications",
]

EDUCATION_MAP = {0: "High School", 1: "Bachelor's", 2: "Master's", 3: "PhD"}


# ─── Main entry point (critical path — no SHAP) ───────────────────────────────

def predict(
    model,
    features:       Dict[str, Any],
    sensitive_attr: Optional[str] = None,
    domain:         str = "hiring",
) -> dict:
    """
    Fast synchronous prediction — SHAP excluded from this path.

    Returns
    -------
    {
        "prediction":     int,    # 1=Hired, 0=Not Hired
        "confidence":     float,
        "shap_values":    {},     # always empty — populated asynchronously
        "shap_available": False,  # updated to True when cache is ready
        "shap_status":    "pending",
        "explanation":    str,    # rule-based (instant)
        "bias_risk":      dict,   # computed without SHAP (Phase-1 weights)
        "input_row":      list,   # forwarded to the background SHAP task
        "feature_names":  list,
    }
    """
    input_row = _build_input_row(features)

    # ── Fast path: predict + proba only ──────────────────────────────────────
    prediction = int(model.predict(input_row)[0])

    confidence = 0.5
    if hasattr(model, "predict_proba"):
        proba      = model.predict_proba(input_row)[0]
        confidence = round(float(proba[1]), 4)

    # ── Bias risk without SHAP (Phase-1 weights kick in automatically) ────────
    bias_risk = compute_bias_risk_score(
        confidence     = confidence,
        shap_values    = None,          # no SHAP yet — component 2 = 0
        sensitive_attr = sensitive_attr,
        domain         = domain,
    )

    explanation = _rule_based_explanation(features, prediction)

    return {
        "prediction":     prediction,
        "confidence":     confidence,
        "shap_values":    {},
        "shap_available": False,
        "shap_status":    "pending",
        "explanation":    explanation,
        "bias_risk":      bias_risk,
        "input_row":      input_row,    # passed to compute_shap_background
        "feature_names":  FEATURE_NAMES,
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
        return f"Hired — candidate shows {reason}. (Full SHAP explanation pending)"

    weaknesses = []
    if tech < 50: weaknesses.append(f"low technical score ({tech}/100)")
    if comm < 50: weaknesses.append(f"low communication score ({comm}/100)")
    if exp  < 2:  weaknesses.append(f"limited experience ({exp} years)")
    if features.get("education_level", 1) == 0:
        weaknesses.append("no degree beyond high school")
    reason = ", ".join(weaknesses) or "did not meet minimum requirements"
    return f"Not hired — {reason}. (Full SHAP explanation pending)"
