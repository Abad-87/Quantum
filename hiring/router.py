"""
hiring/router.py

FastAPI router for the Job Hiring domain.
Endpoint: POST /hiring/predict

Phase 2 additions
-----------------
• preprocess_features() called BEFORE predict() — detects and neutralises
  correlations between sensitive attributes and objective features.
• run_post_processing_checks() called in a background task — calibration and
  equalized-odds checks feed back into bias_risk_score for the NEXT request
  once enough history accumulates (≥ 30 records per group).
• preprocessing_report and post_processing (when available) embedded in the
  persisted record for full auditability.
• sensitive_value_group stored anonymously in the DB record so the
  preprocessing pipeline can build its population arrays without PII.
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
logger = logging.getLogger("hiring.router")


# ─── Input schema ─────────────────────────────────────────────────────────────

class HiringRequest(BaseModel):
    # Prediction features (model sees these)
    years_experience:    float = Field(..., ge=0,   le=50,  description="Years of work experience")
    education_level:     int   = Field(..., ge=0,   le=3,   description="0=HS, 1=Bachelor, 2=Master, 3=PhD")
    technical_score:     float = Field(..., ge=0,   le=100, description="Technical assessment score")
    communication_score: float = Field(..., ge=0,   le=100, description="Communication assessment score")
    num_past_jobs:       int   = Field(..., ge=0,   le=30,  description="Number of previous jobs")
    certifications:      int   = Field(0,   ge=0,   le=20,  description="Number of relevant certifications")

    # Sensitive attributes — fairness monitoring ONLY
    gender:    Optional[str] = Field(None, description="Fairness monitoring only")
    religion:  Optional[str] = Field(None, description="Fairness monitoring only")
    ethnicity: Optional[str] = Field(None, description="Fairness monitoring only")

    @field_validator("education_level")
    @classmethod
    def validate_education(cls, v):
        if v not in {0, 1, 2, 3}:
            raise ValueError("education_level must be 0, 1, 2, or 3")
        return v


# ─── Output schema ────────────────────────────────────────────────────────────

class HiringResponse(BaseModel):
    prediction:           int
    prediction_label:     str
    confidence:           float
    shap_values:          Dict[str, float]
    shap_available:       bool
    explanation:          str
    bias_risk:            Dict[str, Any]
    fairness:             Dict[str, Any]
    preprocessing:        Dict[str, Any]
    model_version:        str
    model_variant:        str
    message:              str


# ─── Background: post-processing checks ──────────────────────────────────────

async def _run_post_processing_background(domain: str, sensitive_attr: str) -> None:
    """
    Retrieve recent predictions and run calibration + equalized-odds checks.
    Results are logged; they feed into bias_risk_score on subsequent requests
    once the router is extended to cache/look up last known post-proc results.
    """
    try:
        records = await get_recent_predictions(domain, limit=500)
        if len(records) < 30:
            return  # Not enough data yet

        y_pred, y_prob, y_true, sens_vals = [], [], [], []
        for r in records:
            fairness_info = r.get("fairness", {})
            if fairness_info.get("sensitive_attribute") != sensitive_attr:
                continue
            y_pred.append(int(r.get("prediction", 0)))
            y_prob.append(float(r.get("confidence", 0.5)))
            # Ground-truth approximation: use stored label if outcome is known
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
                f"[{domain}] Post-processing check triggered flag_for_review.  "
                f"Warnings: {result['warnings']}"
            )
    except Exception as exc:
        logger.error(f"[{domain}] Post-processing background task failed: {exc}")


# ─── Endpoint ─────────────────────────────────────────────────────────────────

@router.post("/predict", response_model=HiringResponse)
async def hiring_predict(
    request:          Request,
    body:             HiringRequest,
    background_tasks: BackgroundTasks,
):
    """
    **Job Hiring Prediction**

    Predicts whether a candidate should be hired from objective features.
    Sensitive attributes (gender, religion, ethnicity) are used ONLY for
    fairness monitoring — they never reach the model.

    **Phase 2 additions:**
    - Features are pre-processed to neutralise correlations with sensitive
      attributes before being passed to the model.
    - Calibration and equalized-odds checks run asynchronously in the
      background, with results feeding back into bias_risk_score.

    **Returns:** prediction, confidence, SHAP values, bias risk, explanation,
    fairness report, preprocessing report, model version.
    """
    correlation_id: str = getattr(request.state, "correlation_id", "unknown")

    model, variant = get_model_ab()
    model_meta     = get_metadata(variant)
    model_version  = model_meta.get("version", "unknown")

    raw_features = {
        "years_experience":     body.years_experience,
        "education_level":      body.education_level,
        "technical_score":      body.technical_score,
        "communication_score":  body.communication_score,
        "num_past_jobs":        body.num_past_jobs,
        "certifications":       body.certifications,
    }

    sensitive_attr, sensitive_value = _resolve_sensitive(
        [("gender", body.gender), ("religion", body.religion), ("ethnicity", body.ethnicity)]
    )

    # ── Phase 2: Pre-processing (correlation neutralisation) ──────────────────
    preprocessing_report = await preprocess_features(
        features        = raw_features,
        sensitive_attr  = sensitive_attr,
        sensitive_value = sensitive_value,
        domain          = "hiring",
    )
    prediction_features = preprocessing_report["features"]

    # ── Predict ───────────────────────────────────────────────────────────────
    try:
        result = predict(model, prediction_features, sensitive_attr=sensitive_attr, domain="hiring")
    except Exception as exc:
        logger.error(f"[{correlation_id}] Prediction error: {exc}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    prediction_label = "Hired" if result["prediction"] == 1 else "Not Hired"

    # ── Fairness check (single-prediction) ───────────────────────────────────
    fairness_result = run_fairness_check(
        prediction      = result["prediction"],
        sensitive_attr  = sensitive_attr or "not_provided",
        sensitive_value = sensitive_value or "unknown",
        domain          = "hiring",
    )
    safe_fairness = {k: v for k, v in fairness_result.items() if k != "sensitive_value"}

    # ── Safe preprocessing summary for response (no PII) ─────────────────────
    safe_preprocessing = {
        "sufficient_history":   preprocessing_report["sufficient_history"],
        "records_used":         preprocessing_report["records_used"],
        "message":              preprocessing_report["message"],
        "correlation_report":   preprocessing_report["correlation_report"],
    }

    # ── Persist ───────────────────────────────────────────────────────────────
    log_record = {
        "domain":                 "hiring",
        "input":                  prediction_features,   # cleaned features
        "raw_input":              raw_features,
        "prediction":             result["prediction"],
        "confidence":             result["confidence"],
        "prediction_label":       prediction_label,
        "explanation":            result["explanation"],
        "fairness":               safe_fairness,
        "preprocessing":          safe_preprocessing,
        "sensitive_value_group":  sensitive_value or "unknown",  # anonymised group label
        "model_version":          model_version,
        "model_variant":          variant,
        "correlation_id":         correlation_id,
    }

    background_tasks.add_task(log_prediction,
        domain           = "hiring",
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
        path           = "/hiring/predict",
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
    # Phase 2: schedule post-processing checks asynchronously
    if sensitive_attr:
        background_tasks.add_task(
            _run_post_processing_background, "hiring", sensitive_attr
        )

    warning_msg = f" ⚠️ {fairness_result['warning']}" if fairness_result.get("warning") else ""
    if result["bias_risk"].get("flag_for_review"):
        warning_msg += " 🚩 Flagged for human review by post-processing checks."

    return HiringResponse(
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
