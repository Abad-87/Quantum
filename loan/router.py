"""
loan/router.py

FastAPI router for Banking & Loan Approval.
Endpoint: POST /loan/predict

Phase 2 additions: preprocess_features + run_post_processing_checks.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel, Field, field_validator
from typing import Any, Dict, Optional
import logging

from .model_loader import get_model_ab, get_metadata
from .predictor import predict
from fairness.checker import run_fairness_check, run_post_processing_checks
from utils.logger import log_prediction, log_correlation_event
from utils.database import save_prediction, preprocess_features, get_recent_predictions

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
    preprocessing:    Dict[str, Any]
    model_version:    str
    model_variant:    str
    message:          str


# ─── Background: post-processing checks ──────────────────────────────────────

async def _run_post_processing_background(domain: str, sensitive_attr: str) -> None:
    try:
        records = await get_recent_predictions(domain, limit=500)
        if len(records) < 30:
            return

        y_pred, y_prob, y_true, sens_vals = [], [], [], []
        for r in records:
            if r.get("fairness", {}).get("sensitive_attribute") != sensitive_attr:
                continue
            y_pred.append(int(r.get("prediction", 0)))
            y_prob.append(float(r.get("confidence", 0.5)))
            y_true.append(int(r.get("ground_truth", r.get("prediction", 0))))
            sens_vals.append(str(r.get("sensitive_value_group", "unknown")))

        if len(y_pred) < 30:
            return

        result = run_post_processing_checks(
            y_pred=y_pred, y_prob=y_prob, y_true=y_true,
            sensitive_values=sens_vals,
            sensitive_attr=sensitive_attr,
            domain=domain,
        )
        if result["flag_for_review"]:
            logger.warning(
                f"[{domain}] Post-processing flag_for_review=True  "
                f"Warnings: {result['warnings']}"
            )
    except Exception as exc:
        logger.error(f"[{domain}] Post-processing background task failed: {exc}")


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

    **Phase 2:** Features pre-processed to neutralise correlation with sensitive
    attributes. Calibration and equalized-odds checks run in the background.

    **Returns:** approval decision, confidence, SHAP values, bias risk,
    explanation, fairness report, preprocessing report, model version.
    """
    correlation_id: str = getattr(request.state, "correlation_id", "unknown")

    model, variant = get_model_ab()
    model_meta     = get_metadata(variant)
    model_version  = model_meta.get("version", "unknown")

    raw_features = {
        "credit_score":     body.credit_score,
        "annual_income":    body.annual_income,
        "loan_amount":      body.loan_amount,
        "loan_term_months": body.loan_term_months,
        "employment_years": body.employment_years,
        "existing_debt":    body.existing_debt,
        "num_credit_lines": body.num_credit_lines,
    }

    sensitive_attr, sensitive_value = _resolve_sensitive([
        ("gender",    body.gender),
        ("religion",  body.religion),
        ("ethnicity", body.ethnicity),
        ("age_group", body.age_group),
    ])

    # ── Phase 2: Pre-processing (correlation neutralisation) ──────────────────
    preprocessing_report = await preprocess_features(
        features        = raw_features,
        sensitive_attr  = sensitive_attr,
        sensitive_value = sensitive_value,
        domain          = "loan",
    )
    prediction_features = preprocessing_report["features"]

    # ── Predict ───────────────────────────────────────────────────────────────
    try:
        result = predict(
            model, prediction_features,
            sensitive_attr=sensitive_attr, domain="loan"
        )
    except Exception as exc:
        logger.error(f"[{correlation_id}] Loan prediction error: {exc}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    prediction_label = "Approved" if result["prediction"] == 1 else "Rejected"

    # ── Fairness check (single-prediction) ───────────────────────────────────
    fairness_result = run_fairness_check(
        prediction      = result["prediction"],
        sensitive_attr  = sensitive_attr or "not_provided",
        sensitive_value = sensitive_value or "unknown",
        domain          = "loan",
    )
    safe_fairness = {k: v for k, v in fairness_result.items() if k != "sensitive_value"}

    safe_preprocessing = {
        "sufficient_history": preprocessing_report["sufficient_history"],
        "records_used":       preprocessing_report["records_used"],
        "message":            preprocessing_report["message"],
        "correlation_report": preprocessing_report["correlation_report"],
    }

    # ── Persist ───────────────────────────────────────────────────────────────
    log_record = {
        "domain":                "loan",
        "input":                 prediction_features,
        "raw_input":             raw_features,
        "prediction":            result["prediction"],
        "confidence":            result["confidence"],
        "prediction_label":      prediction_label,
        "explanation":           result["explanation"],
        "fairness":              safe_fairness,
        "preprocessing":         safe_preprocessing,
        "sensitive_value_group": sensitive_value or "unknown",
        "model_version":         model_version,
        "model_variant":         variant,
        "correlation_id":        correlation_id,
    }
    background_tasks.add_task(log_prediction,
        domain           = "loan",
        input_data       = prediction_features,
        prediction       = result["prediction"],
        prediction_label = prediction_label,
        explanation      = result["explanation"],
        fairness_result  = fairness_result,
        correlation_id   = correlation_id,
    )
    background_tasks.add_task(save_prediction, log_record)
    background_tasks.add_task(
        log_correlation_event,
        correlation_id = correlation_id,
        event          = "prediction_complete",
        path           = "/loan/predict",
        method         = "POST",
        model_metadata = model_meta,
        result         = {
            "prediction":        result["prediction"],
            "prediction_label":  prediction_label,
            "confidence":        result["confidence"],
            "bias_risk_score":   result["bias_risk"]["score"],
            "bias_risk_band":    result["bias_risk"]["band"],
            "flag_for_review":   result["bias_risk"]["flag_for_review"],
        },
    )
    if sensitive_attr:
        background_tasks.add_task(
            _run_post_processing_background, "loan", sensitive_attr
        )

    warning_msg = f" ⚠️ {fairness_result['warning']}" if fairness_result.get("warning") else ""
    if result["bias_risk"].get("flag_for_review"):
        warning_msg += " 🚩 Flagged for human review by post-processing checks."

    return LoanResponse(
        prediction       = result["prediction"],
        prediction_label = prediction_label,
        confidence       = result["confidence"],
        shap_values      = result["shap_values"],
        shap_available   = result["shap_available"],
        explanation      = result["explanation"],
        bias_risk        = result["bias_risk"],
        fairness         = safe_fairness,
        preprocessing    = safe_preprocessing,
        model_version    = model_version,
        model_variant    = variant,
        message          = f"Prediction complete.{warning_msg}",
    )


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _resolve_sensitive(pairs: list) -> tuple:
    """Return (attr_name, value) for the first non-None sensitive attribute."""
    for attr, val in pairs:
        if val:
            return attr, val
    return None, None
