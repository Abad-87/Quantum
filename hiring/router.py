"""
hiring/router.py  —  Phase 3: async SHAP

Phase 3 changes
---------------
• predict() no longer blocks on SHAP — returns immediately with rule-based
  explanation and shap_status: "pending".
• compute_shap_background() is added as a BackgroundTask so the event loop
  is never blocked by the TreeExplainer CPU work.
• Response now includes shap_status ("pending") and a shap_poll_url hint
  so clients know where to retrieve the full explanation.
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
from utils.shap_cache import compute_shap_background

router = APIRouter()
logger = logging.getLogger("hiring.router")


class HiringRequest(BaseModel):
    years_experience:    float = Field(..., ge=0,   le=50)
    education_level:     int   = Field(..., ge=0,   le=3)
    technical_score:     float = Field(..., ge=0,   le=100)
    communication_score: float = Field(..., ge=0,   le=100)
    num_past_jobs:       int   = Field(..., ge=0,   le=30)
    certifications:      int   = Field(0,   ge=0,   le=20)
    gender:    Optional[str]   = Field(None)
    religion:  Optional[str]   = Field(None)
    ethnicity: Optional[str]   = Field(None)

    @field_validator("education_level")
    @classmethod
    def validate_education(cls, v):
        if v not in {0, 1, 2, 3}:
            raise ValueError("education_level must be 0, 1, 2, or 3")
        return v


class HiringResponse(BaseModel):
    prediction:        int
    prediction_label:  str
    confidence:        float
    shap_values:       Dict[str, float]   # {} on first response
    shap_available:    bool               # False on first response
    shap_status:       str                # "pending" | "ready" | "error"
    shap_poll_url:     str                # GET /shap/{correlation_id}
    explanation:       str                # rule-based (instant)
    bias_risk:         Dict[str, Any]
    fairness:          Dict[str, Any]
    preprocessing:     Dict[str, Any]
    model_version:     str
    model_variant:     str
    correlation_id:    str
    message:           str


async def _run_post_processing_background(domain: str, sensitive_attr: str) -> None:
    try:
        records = await get_recent_predictions(domain, limit=500, sensitive_attr=sensitive_attr)
        if len(records) < 30:
            return
        y_pred, y_prob, y_true, sens_vals = [], [], [], []
        for r in records:
            y_pred.append(int(r.get("prediction", 0)))
            y_prob.append(float(r.get("confidence", 0.5)))
            y_true.append(int(r.get("ground_truth", r.get("prediction", 0))))
            sens_vals.append(str(r.get("sensitive_value_group", "unknown")))
        if len(y_pred) < 30:
            return
        result = run_post_processing_checks(
            y_pred=y_pred, y_prob=y_prob, y_true=y_true,
            sensitive_values=sens_vals, sensitive_attr=sensitive_attr, domain=domain,
        )
        if result["flag_for_review"]:
            logger.warning(f"[{domain}] flag_for_review=True  {result['warnings']}")
    except Exception as exc:
        logger.error(f"[{domain}] post-processing background failed: {exc}")


@router.post("/predict", response_model=HiringResponse)
async def hiring_predict(
    request:          Request,
    body:             HiringRequest,
    background_tasks: BackgroundTasks,
):
    """
    **Job Hiring Prediction**

    Returns the decision immediately.  Full SHAP explanation is computed
    asynchronously and available within seconds at:
    - **GET** `/shap/{correlation_id}` — REST poll
    - **WS**  `/shap/ws/{correlation_id}` — WebSocket push
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

    sensitive_attr, sensitive_value = _resolve_sensitive([
        ("gender", body.gender), ("religion", body.religion), ("ethnicity", body.ethnicity),
    ])

    # ── Phase 2: pre-processing ───────────────────────────────────────────────
    preprocessing_report = await preprocess_features(
        features=raw_features, sensitive_attr=sensitive_attr,
        sensitive_value=sensitive_value, domain="hiring",
    )
    prediction_features = preprocessing_report["features"]

    # ── Phase 3: fast predict (no SHAP) ──────────────────────────────────────
    try:
        result = predict(model, prediction_features, sensitive_attr=sensitive_attr, domain="hiring")
    except Exception as exc:
        logger.error(f"[{correlation_id}] Prediction error: {exc}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    prediction_label = "Hired" if result["prediction"] == 1 else "Not Hired"

    fairness_result = run_fairness_check(
        prediction=result["prediction"],
        sensitive_attr=sensitive_attr or "not_provided",
        sensitive_value=sensitive_value or "unknown",
        domain="hiring",
    )
    safe_fairness = {k: v for k, v in fairness_result.items() if k != "sensitive_value"}

    safe_preprocessing = {
        "sufficient_history": preprocessing_report["sufficient_history"],
        "records_used":       preprocessing_report["records_used"],
        "message":            preprocessing_report["message"],
        "correlation_report": preprocessing_report["correlation_report"],
    }

    log_record = {
        "domain": "hiring", "input": prediction_features, "raw_input": raw_features,
        "prediction": result["prediction"], "confidence": result["confidence"],
        "prediction_label": prediction_label, "explanation": result["explanation"],
        "fairness": safe_fairness, "preprocessing": safe_preprocessing,
        "sensitive_value_group": sensitive_value or "unknown",
        "model_version": model_version, "model_variant": variant,
        "correlation_id": correlation_id,
    }

    background_tasks.add_task(log_prediction,
        domain=  "hiring", input_data=prediction_features,
        prediction=result["prediction"], prediction_label=prediction_label,
        explanation=result["explanation"], fairness_result=fairness_result,
        correlation_id=correlation_id,
    )
    background_tasks.add_task(save_prediction, log_record)
    background_tasks.add_task(log_correlation_event,
        correlation_id=correlation_id, event="prediction_complete",
        path="/hiring/predict", method="POST", model_metadata=model_meta,
        result={
            "prediction": result["prediction"], "prediction_label": prediction_label,
            "confidence": result["confidence"],
            "bias_risk_score": result["bias_risk"]["score"],
            "bias_risk_band":  result["bias_risk"]["band"],
            "flag_for_review": result["bias_risk"]["flag_for_review"],
            "shap_status": "pending",
        },
    )

    # ── Phase 3: schedule async SHAP computation ──────────────────────────────
    background_tasks.add_task(
        compute_shap_background,
        model, result["input_row"], result["prediction"],
        result["feature_names"], correlation_id, "hiring",
        prediction_features, sensitive_attr,
    )

    if sensitive_attr:
        background_tasks.add_task(_run_post_processing_background, "hiring", sensitive_attr)

    warning_msg = f" ⚠️ {fairness_result['warning']}" if fairness_result.get("warning") else ""
    if result["bias_risk"].get("flag_for_review"):
        warning_msg += " 🚩 Flagged for human review."

    return HiringResponse(
        prediction       = result["prediction"],
        prediction_label = prediction_label,
        confidence       = result["confidence"],
        shap_values      = result["shap_values"],
        shap_available   = result["shap_available"],
        shap_status      = result["shap_status"],
        shap_poll_url    = f"/shap/{correlation_id}",
        explanation      = result["explanation"],
        bias_risk        = result["bias_risk"],
        fairness         = safe_fairness,
        preprocessing    = safe_preprocessing,
        model_version    = model_version,
        model_variant    = variant,
        correlation_id   = correlation_id,
        message          = f"Prediction complete. SHAP computing asynchronously.{warning_msg}",
    )


def _resolve_sensitive(pairs: list) -> tuple:
    for attr, val in pairs:
        if val:
            return attr, val
    return None, None
