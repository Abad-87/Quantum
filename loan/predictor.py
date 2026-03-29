"""
loan/predictor.py

Prediction logic for Banking & Loan Approval domain.

Model features (no sensitive attributes):
- credit_score      : 300–850
- annual_income     : in USD
- loan_amount       : requested amount in USD
- loan_term_months  : repayment period (12, 24, 36, 60, etc.)
- employment_years  : years at current employer
- existing_debt     : current outstanding debt in USD
- num_credit_lines  : number of open credit accounts
"""

import numpy as np
import logging

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

# Derived ratio for explainability
def _debt_to_income(existing_debt, annual_income):
    if annual_income == 0:
        return 999
    return round(existing_debt / annual_income, 3)


def _loan_to_income(loan_amount, annual_income):
    if annual_income == 0:
        return 999
    return round(loan_amount / annual_income, 3)


def predict(model, features: dict) -> tuple[int, float, str]:
    """
    Returns (prediction, confidence, explanation).
    prediction: 1 = Approved, 0 = Rejected
    """
    input_row = [[
        features["credit_score"],
        features["annual_income"],
        features["loan_amount"],
        features["loan_term_months"],
        features["employment_years"],
        features["existing_debt"],
        features["num_credit_lines"],
    ]]

    prediction = int(model.predict(input_row)[0])

    confidence = 0.5
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_row)[0]
        confidence = round(float(proba[1]), 3)

    explanation = _explain(model, features, prediction, input_row)
    return prediction, confidence, explanation


def _explain(model, features: dict, prediction: int, input_row: list) -> str:
    try:
        import shap
        base_model = model
        if hasattr(model, "steps"):
            base_model = model.steps[-1][1]

        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(input_row)

        if isinstance(shap_values, list):
            values = shap_values[prediction]
        else:
            values = shap_values[0]

        shap_pairs = sorted(
            zip(FEATURE_NAMES, values[0] if hasattr(values[0], "__iter__") else values),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        top_feature, top_value = shap_pairs[0]
        pretty = top_feature.replace("_", " ")
        direction = "high" if top_value > 0 else "low"

        if prediction == 1:
            return f"Loan approved — primarily driven by {direction} {pretty} ({features.get(top_feature)})."
        else:
            return f"Loan rejected — primarily due to {direction} {pretty} ({features.get(top_feature)})."

    except Exception as e:
        logger.debug(f"SHAP unavailable ({e}). Using rule-based explanation.")
        return _rule_based_explanation(features, prediction)


def _rule_based_explanation(features: dict, prediction: int) -> str:
    credit    = features.get("credit_score", 0)
    income    = features.get("annual_income", 1)
    debt      = features.get("existing_debt", 0)
    loan_amt  = features.get("loan_amount", 0)
    emp_years = features.get("employment_years", 0)

    dti   = _debt_to_income(debt, income)
    lti   = _loan_to_income(loan_amt, income)

    if prediction == 1:
        factors = []
        if credit >= 700:
            factors.append(f"good credit score ({credit})")
        if dti < 0.4:
            factors.append(f"manageable debt-to-income ratio ({dti:.0%})")
        if emp_years >= 2:
            factors.append(f"{emp_years} years of stable employment")
        reason = ", ".join(factors) if factors else "meets all lending criteria"
        return f"Loan approved — {reason}."
    else:
        issues = []
        if credit < 600:
            issues.append(f"low credit score ({credit}, minimum 600 recommended)")
        if dti > 0.5:
            issues.append(f"high debt-to-income ratio ({dti:.0%})")
        if lti > 3:
            issues.append(f"loan amount too high relative to income (ratio: {lti:.1f}x)")
        if emp_years < 1:
            issues.append("less than 1 year of employment history")
        reason = "; ".join(issues) if issues else "does not meet lending criteria"
        return f"Loan rejected — {reason}."
