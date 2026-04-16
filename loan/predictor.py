"""
loan/predictor.py  —  Phase 3: async SHAP

Critical-path changes: SHAP removed.  See hiring/predictor.py for full notes.
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


def predict(
    model,
    features:       Dict[str, Any],
    sensitive_attr: Optional[str] = None,
    domain:         str = "loan",
) -> dict:
    """Fast synchronous prediction — SHAP computed asynchronously."""
    input_row = _build_input_row(features)

    prediction = int(model.predict(input_row)[0])

    confidence = 0.5
    if hasattr(model, "predict_proba"):
        proba      = model.predict_proba(input_row)[0]
        confidence = round(float(proba[1]), 4)

    bias_risk = compute_bias_risk_score(
        confidence     = confidence,
        shap_values    = None,
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
        "input_row":      input_row,
        "feature_names":  FEATURE_NAMES,
    }


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
        return f"Loan approved — {reason}. (Full SHAP explanation pending)"

    issues = []
    if credit    < 600: issues.append(f"low credit score ({credit}, minimum 600 recommended)")
    if dti       > 0.5: issues.append(f"high debt-to-income ratio ({dti:.0%})")
    if lti       > 3:   issues.append(f"loan amount too high relative to income ({lti:.1f}x income)")
    if emp_years  < 1:  issues.append("less than 1 year of employment history")
    reason = "; ".join(issues) or "does not meet lending criteria"
    return f"Loan rejected — {reason}. (Full SHAP explanation pending)"
