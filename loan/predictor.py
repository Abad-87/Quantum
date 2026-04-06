"""
loan/predictor.py

Prediction logic for the Banking & Loan Approval domain.

predict() returns a structured dict:
{
    "prediction":      int,    # 1 = Approved, 0 = Rejected
    "confidence":      float,
    "shap_values":     dict,   # {feature_name: float}
    "shap_available":  bool,
    "explanation":     str,
    "bias_risk":       dict,   # from compute_bias_risk_score()
}
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fairness.checker import compute_bias_risk_score

logger = logging.getLogger("loan.predictor")

FEATURE_NAMES = [
    "credit_score",
    "annual_income",
    "loan_amount",
    "loan_term_months",
    "employment_years",
    "existing_debt",
    "num_credit_lines",
]


# ─── Main entry point ─────────────────────────────────────────────────────────

def predict(
    model,
    features: Dict[str, Any],
    sensitive_attr: Optional[str] = None,
    domain: str = "loan",
) -> dict:
    """
    Run loan-approval prediction and return a fully structured result dict.

    Parameters
    ----------
    model          : Loaded sklearn model/pipeline from the registry.
    features       : Validated financial features — no sensitive attributes.
    sensitive_attr : Sensitive attribute name for bias-risk weighting only.
    domain         : Domain label forwarded to bias-risk computation.
    """
    input_row = _build_input_row(features)

    # ── Prediction ────────────────────────────────────────────────────────────
    prediction = int(model.predict(input_row)[0])

    # ── Confidence ────────────────────────────────────────────────────────────
    confidence = 0.5
    if hasattr(model, "predict_proba"):
        proba      = model.predict_proba(input_row)[0]
        confidence = round(float(proba[1]), 4)

    # ── SHAP values ───────────────────────────────────────────────────────────
    shap_values, shap_available = _compute_shap(model, input_row, prediction)

    # ── Bias risk ─────────────────────────────────────────────────────────────
    bias_risk = compute_bias_risk_score(
        confidence=confidence,
        shap_values=shap_values,
        sensitive_attr=sensitive_attr,
        domain=domain,
    )

    # ── Explanation ───────────────────────────────────────────────────────────
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
        features["credit_score"],
        features["annual_income"],
        features["loan_amount"],
        features["loan_term_months"],
        features["employment_years"],
        features["existing_debt"],
        features["num_credit_lines"],
    ]]


def _compute_shap(
    model,
    input_row: list,
    prediction: int,
) -> tuple[Dict[str, float], bool]:
    try:
        import shap

        base_model = model.steps[-1][1] if hasattr(model, "steps") else model
        explainer  = shap.TreeExplainer(base_model)
        raw        = explainer.shap_values(input_row)

        values = raw[prediction] if isinstance(raw, list) else raw
        flat   = values[0] if hasattr(values[0], "__iter__") else values

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
        return f"Loan approved — primarily driven by {direction} {pretty} ({feat_value})."
    return f"Loan rejected — primarily due to {direction} {pretty} ({feat_value})."


def _rule_based_explanation(features: dict, prediction: int) -> str:
    credit    = features.get("credit_score", 0)
    income    = features.get("annual_income", 1) or 1
    debt      = features.get("existing_debt", 0)
    loan_amt  = features.get("loan_amount", 0)
    emp_years = features.get("employment_years", 0)

    dti = round(debt / income, 3)
    lti = round(loan_amt / income, 3)

    if prediction == 1:
        factors = []
        if credit    >= 700: factors.append(f"good credit score ({credit})")
        if dti        < 0.4: factors.append(f"manageable debt-to-income ratio ({dti:.0%})")
        if emp_years  >= 2:  factors.append(f"{emp_years} years of stable employment")
        reason = ", ".join(factors) or "meets all lending criteria"
        return f"Loan approved — {reason}."

    issues = []
    if credit    < 600: issues.append(f"low credit score ({credit}, minimum 600 recommended)")
    if dti       > 0.5: issues.append(f"high debt-to-income ratio ({dti:.0%})")
    if lti       > 3:   issues.append(f"loan amount too high relative to income ({lti:.1f}x income)")
    if emp_years  < 1:  issues.append("less than 1 year of employment history")
    reason = "; ".join(issues) or "does not meet lending criteria"
    return f"Loan rejected — {reason}."
