"""
social/router.py

FastAPI router for Social Media Recommendation.
Endpoint: POST /social/recommend

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
logger = logging.getLogger("social.router")


# ─── Input schema ─────────────────────────────────────────────────────────────

class SocialRequest(BaseModel):
    avg_session_minutes: float = Field(..., ge=0,   le=1440, description="Avg daily session length (minutes)")
    posts_per_day:       float = Field(0.0, ge=0,   le=100,  description="Avg posts per day")
    topics_interacted:   int   = Field(..., ge=0,   le=50,   description="Distinct topics interacted with")
    like_rate:           float = Field(..., ge=0.0, le=1.0,  description="Ratio of liked to seen content")
    share_rate:          float = Field(0.0, ge=0.0, le=1.0,  description="Ratio of shared to liked content")
    comment_rate:        float = Field(0.0, ge=0.0, le=1.0,  description="Ratio of commented to seen content")
    account_age_days:    int   = Field(..., ge=0,   le=10000,description="Account age in days")

    # Sensitive attributes — fairness monitoring ONLY
    gender:    Optional[str] = Field(None, description="Fairness monitoring only")
    age_group: Optional[str] = Field(None, description="Fairness monitoring only")
    location:  Optional[str] = Field(None, description="Fairness monitoring only")
    language:  Optional[str] = Field(None, description="Fairness monitoring only")

    @field_validator("like_rate", "share_rate", "comment_rate")
    @classmethod
    def rate_in_range(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Rate values must be between 0.0 and 1.0")
        return v


# ─── Output schema ────────────────────────────────────────────────────────────

class SocialResponse(BaseModel):
    recommended_category_id: int
    recommended_category:    str
    confidence:              float
    shap_values:             Dict[str, float]
    shap_available:          bool
    explanation:             str
    bias_risk:               Dict[str, Any]
    fairness:                Dict[str, Any]
    preprocessing:           Dict[str, Any]
    model_version:           str
    model_variant:           str
    message:                 str


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

@router.post("/recommend", response_model=SocialResponse)
async def social_recommend(
    request:          Request,
    body:             SocialRequest,
    background_tasks: BackgroundTasks,
):
    """
    **Social Media Content Recommendation**

    Recommends a content category from behavioural signals only.
    Demographic attributes are used ONLY for fairness auditing —
    never to drive the recommendation (prevents demographic filter bubbles).

    **Phase 2:** Features pre-processed to neutralise correlation with sensitive
    attributes. Calibration and equalized-odds checks run in the background.

    **Returns:** category, confidence, SHAP values, bias risk, explanation,
    fairness report, preprocessing report, model version.
    """
    correlation_id: str = getattr(request.state, "correlation_id", "unknown")

    model, variant = get_model_ab()
    model_meta     = get_metadata(variant)
    model_version  = model_meta.get("version", "unknown")

    raw_features = {
        "avg_session_minutes": body.avg_session_minutes,
        "posts_per_day":       body.posts_per_day,
        "topics_interacted":   body.topics_interacted,
        "like_rate":           body.like_rate,
        "share_rate":          body.share_rate,
        "comment_rate":        body.comment_rate,
        "account_age_days":    body.account_age_days,
    }

    sensitive_attr, sensitive_value = _resolve_sensitive([
        ("gender",    body.gender),
        ("age_group", body.age_group),
        ("location",  body.location),
        ("language",  body.language),
    ])

    # ── Phase 2: Pre-processing (correlation neutralisation) ──────────────────
    preprocessing_report = await preprocess_features(
        features        = raw_features,
        sensitive_attr  = sensitive_attr,
        sensitive_value = sensitive_value,
        domain          = "social",
    )
    prediction_features = preprocessing_report["features"]

    # ── Predict ───────────────────────────────────────────────────────────────
    try:
        result = predict(
            model, prediction_features,
            sensitive_attr=sensitive_attr, domain="social"
        )
    except Exception as exc:
        logger.error(f"[{correlation_id}] Social prediction error: {exc}")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {exc}")

    # ── Fairness check (single-prediction) ───────────────────────────────────
    fairness_result = run_fairness_check(
        prediction      = result["prediction"],
        sensitive_attr  = sensitive_attr or "not_provided",
        sensitive_value = sensitive_value or "unknown",
        domain          = "social",
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
        "domain":                "social",
        "input":                 prediction_features,
        "raw_input":             raw_features,
        "prediction":            result["prediction"],
        "confidence":            result["confidence"],
        "prediction_label":      result["category_label"],
        "explanation":           result["explanation"],
        "fairness":              safe_fairness,
        "preprocessing":         safe_preprocessing,
        "sensitive_value_group": sensitive_value or "unknown",
        "model_version":         model_version,
        "model_variant":         variant,
        "correlation_id":        correlation_id,
    }
    background_tasks.add_task(log_prediction,
        domain           = "social",
        input_data       = prediction_features,
        prediction       = result["prediction"],
        prediction_label = result["category_label"],
        explanation      = result["explanation"],
        fairness_result  = fairness_result,
        correlation_id   = correlation_id,
    )
    background_tasks.add_task(save_prediction, log_record)
    background_tasks.add_task(
        log_correlation_event,
        correlation_id = correlation_id,
        event          = "prediction_complete",
        path           = "/social/recommend",
        method         = "POST",
        model_metadata = model_meta,
        result         = {
            "prediction":        result["prediction"],
            "prediction_label":  result["category_label"],
            "confidence":        result["confidence"],
            "bias_risk_score":   result["bias_risk"]["score"],
            "bias_risk_band":    result["bias_risk"]["band"],
            "flag_for_review":   result["bias_risk"]["flag_for_review"],
        },
    )
    if sensitive_attr:
        background_tasks.add_task(
            _run_post_processing_background, "social", sensitive_attr
        )

    warning_msg = f" ⚠️ {fairness_result['warning']}" if fairness_result.get("warning") else ""
    if result["bias_risk"].get("flag_for_review"):
        warning_msg += " 🚩 Flagged for human review by post-processing checks."

    return SocialResponse(
        recommended_category_id = result["prediction"],
        recommended_category    = result["category_label"],
        confidence              = result["confidence"],
        shap_values             = result["shap_values"],
        shap_available          = result["shap_available"],
        explanation             = result["explanation"],
        bias_risk               = result["bias_risk"],
        fairness                = safe_fairness,
        preprocessing           = safe_preprocessing,
        model_version           = model_version,
        model_variant           = variant,
        message                 = f"Recommendation complete.{warning_msg}",
    )


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _resolve_sensitive(pairs: list) -> tuple:
    """Return (attr_name, value) for the first non-None sensitive attribute."""
    for attr, val in pairs:
        if val:
            return attr, val
    return None, None
