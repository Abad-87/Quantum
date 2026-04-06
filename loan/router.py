"""
loan/router.py

FastAPI router for Banking & Loan Approval.
Endpoint: POST /loan/predict
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel, Field, field_validator
from typing import Any, Dict, Optional
import logging

from .model_loader import get_model_ab, get_metadata
from .predictor import predict
from fairness.checker import run_fairness_check
from utils.logger import log_prediction, log_correlation_event
from utils.database import save_prediction

router = APIRouter()
logger = logging.getLogger("loan.router")


# ─── Input schema ─────────────────────────────────────────────────────────────

class LoanRequest(BaseModel):
    credit_score:      int   = Field(..., ge=300,  le=850,  description="FICO credit score")
    annual_income:     float = Field(..., ge=0,             description="Annual income in USD")
    loan_amount:       float = Field(..., ge=100,           description="Requested loan amount in USD")
    loan_term_months:  int   = Field(..., ge=6,    le=360,  description="Repayment period in months")
    employment_years:  float = Field(..., ge=0,    le=50,   description="Years at current employer")
    existing_debt:     float = Field(0.0, ge=0,            description="Current outstanding debt in USD")
    num_credit_lines:  int   = Field(0,   ge=0,   le=50,   description="Number of open credit accounts")

    # Sensitive attributes — fairness monitoring ONLY
    gender:    Optional[str] = Field(None, description="Fairness monitoring only")
    religion:  Optional[str] = Field(None, description="Fairness monitoring only")
    ethnicity: Optional[str] = Field(None, description="Fairness monitoring only")
    age_group: Optional[str] = Field(None, description="Fairness monitoring only")

    @field_validator("loan_amount")
    @classmethod
    def loan_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("loan_amount must be greater than 0")
        return v


# ─── Output schema ────────────────────────────────────────────────────────────

class LoanResponse(BaseModel):
    prediction:       int
    prediction_label: str
    confidence:       float
    shap_values:      Dict[str, float]
    shap_available:   bool
    explanation:      str
    bias_risk:        Dict[str, Any]
    fairness:         Dict[str, Any]
    model_version:    str
    model_variant:    str
    message:          str


# ─── Endpoint ─────────────────────────────────────────────────────────────────

@router.post("/predict", response_model=LoanResponse)
async def loan_predict(
    request:          Request,
    body:             LoanRequest,
    background_tasks: BackgroundTasks,
):
    """
    **Loan Approval Prediction**

    Evaluates a loan application from financial features only.
    Sensitive attributes are used ONLY for fairness auditing.

    **Returns:** approval decision, confidence, SHAP values, bias risk,
    explanation, fairness report, model version.
    """
    correlation_id: str = getattr(request.state, "correlation_id", "unknown")

    model, variant = get_model_ab()
    model_meta     = get_metadata(variant)
    model_version  = model_meta.get("version", "unknown")

    prediction_features = {
        "credit_score":     body.credit_score,
        "annual_income":    body.annual_income,
        "loan_amount":      body.loan_amount,
        "loan_term_months": body.loan_term_months,
        "employment_years": body.employment_years,
        "existing_debt":    body.existing_debt,
        "num_credit_lines": body.num_credit_lines,
    }

    sensitive_attr  = next(
        (attr for attr, val in [
            ("gender",    body.gender),
            ("religion",  body.religion),
            ("ethnicity", body.ethnicity),
            ("age_group", body.age_group),
        ] if val),
        None,
    )

    try:
        result = predict(
            model,
            prediction_features,
            sensitive_attr=sensitive_attr,
            domain="loan",
        )
    except Exception as exc:
        logger.error(f"[{correlation_id}] Loan prediction error: {exc}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    prediction_label = "Approved" if result["prediction"] == 1 else "Rejected"

    fairness_result = run_fairness_check(
        prediction=result["prediction"],
        sensitive_attr=sensitive_attr or "not_provided",
        sensitive_value=next(
            (v for v in [body.gender, body.religion, body.ethnicity, body.age_group] if v),
            "unknown",
        ),
        domain="loan",
    )
    safe_fairness = {k: v for k, v in fairness_result.items() if k != "sensitive_value"}

    log_record = {
        "domain":           "loan",
        "input":            prediction_features,
        "prediction":       result["prediction"],
        "prediction_label": prediction_label,
        "explanation":      result["explanation"],
        "fairness":         safe_fairness,
        "model_version":    model_version,
        "model_variant":    variant,
        "correlation_id":   correlation_id,
    }
    background_tasks.add_task(log_prediction,
        domain          = "loan",
        input_data      = prediction_features,
        prediction      = result["prediction"],
        prediction_label= prediction_label,
        explanation     = result["explanation"],
        fairness_result = fairness_result,
        correlation_id  = correlation_id,
    )
    background_tasks.add_task(save_prediction, log_record)
    background_tasks.add_task(
        log_correlation_event,
        correlation_id  = correlation_id,
        event           = "prediction_complete",
        path            = "/loan/predict",
        method          = "POST",
        model_metadata  = model_meta,
        result          = {
            "prediction":        result["prediction"],
            "prediction_label":  prediction_label,
            "confidence":        result["confidence"],
            "bias_risk_score":   result["bias_risk"]["score"],
            "bias_risk_band":    result["bias_risk"]["band"],
        },
    )

    warning_msg = f" ⚠️ {fairness_result['warning']}" if fairness_result.get("warning") else ""

    return LoanResponse(
        prediction       = result["prediction"],
        prediction_label = prediction_label,
        confidence       = result["confidence"],
        shap_values      = result["shap_values"],
        shap_available   = result["shap_available"],
        explanation      = result["explanation"],
        bias_risk        = result["bias_risk"],
        fairness         = safe_fairness,
        model_version    = model_version,
        model_variant    = variant,
        message          = f"Prediction complete.{warning_msg}",
    )
