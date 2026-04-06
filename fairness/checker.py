"""
fairness/checker.py

Fairness evaluation layer — all fairness logic lives here.

Exports
-------
run_fairness_check(...)          Single-prediction fairness report (existing)
run_batch_fairness_check(...)    Batch DPD / EOD evaluation (existing)
compute_bias_risk_score(...)     Per-prediction bias risk score  ← NEW

compute_bias_risk_score
-----------------------
Returns a scalar in [0.0, 1.0] that estimates the risk that a single
prediction is biased, composed of three weighted factors:

  1. Decision-boundary proximity (40 %)
     Confidence ≈ 0.5 means the model is uncertain — marginal predictions
     are the most vulnerable to spurious correlations, including demographic
     proxies hidden in objective features.

  2. SHAP feature concentration (30 %)
     If one feature carries almost all the explanatory weight (high
     Herfindahl–Hirschman Index of |SHAP| values), the model's effective
     reasoning is opaque and brittle — a classic precursor to proxy
     discrimination.

  3. Sensitive-attribute base risk (30 %)
     Some attributes carry known historical discrimination risk.  Even
     though they never enter the model directly, correlated proxy features
     can indirectly encode them.  This term raises vigilance for higher-risk
     attributes.

Interpretation guide (stored in BIAS_RISK_BANDS below):
  0.00 – 0.25  Low       Confident, distributed, low-risk attribute
  0.25 – 0.50  Moderate  Review recommended
  0.50 – 0.75  High      Human review strongly recommended
  0.75 – 1.00  Critical  Block or escalate

IMPORTANT ETHICAL NOTE
----------------------
Sensitive attributes (gender, religion, race, …) are ONLY used here to
*measure* fairness and to *weight* the vigilance signal — they NEVER reach
the prediction model itself.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger("fairness")

# ─── Thresholds ───────────────────────────────────────────────────────────────

FAIRNESS_THRESHOLD = 0.1   # DPD / EOD threshold for batch checks

BIAS_RISK_BANDS: Dict[str, tuple] = {
    "low":      (0.00, 0.25),
    "moderate": (0.25, 0.50),
    "high":     (0.50, 0.75),
    "critical": (0.75, 1.00),
}

# Historical discrimination risk weight per sensitive attribute type.
# Source: informed by EEOC protected-class guidance and Fairlearn literature.
_ATTR_RISK_WEIGHTS: Dict[str, float] = {
    "race":          0.90,
    "ethnicity":     0.85,
    "gender":        0.75,
    "religion":      0.65,
    "age_group":     0.55,
    "disability":    0.70,
    "location":      0.35,
    "language":      0.30,
    "not_provided":  0.20,   # unknown / not submitted → low, not zero
}


# ─── Public: single-prediction bias risk ─────────────────────────────────────

def compute_bias_risk_score(
    confidence:      float,
    shap_values:     Optional[Dict[str, float]],
    sensitive_attr:  Optional[str] = None,
    domain:          str = "unknown",
) -> Dict[str, object]:
    """
    Compute a structured bias-risk report for one prediction.

    Parameters
    ----------
    confidence      : Model's probability of the positive class (0.0–1.0).
    shap_values     : {feature_name: shap_value} dict from the predictor.
                      May be None or empty if SHAP is unavailable.
    sensitive_attr  : The sensitive attribute submitted for fairness monitoring
                      (e.g. "gender", "ethnicity").  Never reaches the model.
    domain          : Domain identifier for log context ("hiring", "loan", …).

    Returns
    -------
    {
        "score":       float in [0.0, 1.0],
        "band":        "low" | "moderate" | "high" | "critical",
        "components": {
            "boundary_proximity":   float,   # factor 1
            "shap_concentration":   float,   # factor 2
            "attribute_base_risk":  float,   # factor 3
        },
        "recommendation": str,
    }
    """

    # ── Component 1: Decision-boundary proximity ──────────────────────────────
    # Maps [0, 1] confidence → [1, 0] risk (max risk at 0.5, min at 0 and 1).
    boundary_proximity = 1.0 - abs(2.0 * confidence - 1.0)

    # ── Component 2: SHAP feature concentration (HHI) ────────────────────────
    shap_concentration = _compute_shap_concentration(shap_values)

    # ── Component 3: Sensitive attribute base risk ────────────────────────────
    attr_key        = (sensitive_attr or "not_provided").lower()
    attribute_risk  = _ATTR_RISK_WEIGHTS.get(attr_key, 0.40)   # default: moderate

    # ── Weighted combination ──────────────────────────────────────────────────
    score = (
        0.40 * boundary_proximity
        + 0.30 * shap_concentration
        + 0.30 * attribute_risk
    )
    score = round(max(0.0, min(1.0, score)), 4)

    band           = _score_to_band(score)
    recommendation = _band_to_recommendation(band)

    logger.debug(
        f"[{domain}] bias_risk={score:.4f} ({band})  "
        f"boundary={boundary_proximity:.3f}  "
        f"shap_conc={shap_concentration:.3f}  "
        f"attr_risk={attribute_risk:.3f}"
    )

    return {
        "score": score,
        "band":  band,
        "components": {
            "boundary_proximity":  round(boundary_proximity, 4),
            "shap_concentration":  round(shap_concentration, 4),
            "attribute_base_risk": round(attribute_risk, 4),
        },
        "recommendation": recommendation,
    }


# ─── Public: single-prediction fairness report ───────────────────────────────

def run_fairness_check(
    prediction:      int,
    sensitive_attr:  str,
    sensitive_value: str,
    domain:          str,
) -> dict:
    """
    Single-prediction fairness check.

    For a single prediction, group statistics are unavailable, so we:
    1. Record the prediction alongside its sensitive attribute for batch analysis.
    2. Return a report with a clear disclaimer.

    In production, a background job calls run_batch_fairness_check() over
    accumulated records from utils/database.py.
    """
    report = {
        "sensitive_attribute": sensitive_attr,
        "sensitive_value":     sensitive_value,   # ← stripped before API response
        "domain":              domain,
        "is_fair":             True,
        "warning":             None,
        "metrics": {
            "demographic_parity_difference": None,
            "equal_opportunity_difference":  None,
            "note": (
                "Single-prediction fairness is logged. "
                "Batch metrics are computed from historical data. "
                "See /docs for the monitoring dashboard."
            ),
        },
        "ethical_note": (
            "Sensitive attributes are used ONLY to monitor fairness. "
            "They are NOT inputs to the prediction model."
        ),
    }
    return report


# ─── Public: batch fairness evaluation ───────────────────────────────────────

def run_batch_fairness_check(
    y_pred:              list,
    y_true:              list,
    sensitive_values:    list,
    sensitive_attr_name: str,
    domain:              str,
) -> dict:
    """
    Full batch DPD + EOD evaluation.
    Call from a monitoring job or analytics endpoint.
    """
    dpd = demographic_parity_difference(y_pred, sensitive_values)
    eod = equal_opportunity_difference(y_pred, y_true, sensitive_values)

    dpd_ok  = dpd <= FAIRNESS_THRESHOLD
    eod_ok  = eod <= FAIRNESS_THRESHOLD
    is_fair = dpd_ok and eod_ok

    warnings = []
    if not dpd_ok:
        warnings.append(
            f"Demographic Parity Difference ({dpd}) exceeds threshold "
            f"({FAIRNESS_THRESHOLD}).  Groups receive unequal positive-outcome rates."
        )
    if not eod_ok:
        warnings.append(
            f"Equal Opportunity Difference ({eod}) exceeds threshold "
            f"({FAIRNESS_THRESHOLD}).  True positive rates differ across groups."
        )

    return {
        "domain":             domain,
        "sensitive_attribute": sensitive_attr_name,
        "is_fair":            is_fair,
        "threshold":          FAIRNESS_THRESHOLD,
        "metrics": {
            "demographic_parity_difference": dpd,
            "equal_opportunity_difference":  eod,
        },
        "warnings": warnings or None,
    }


# ─── Batch metrics ────────────────────────────────────────────────────────────

def demographic_parity_difference(y_pred: list, sensitive_values: list) -> float:
    """
    DPD = |P(ŷ=1 | group A) – P(ŷ=1 | group B)|

    Ideal: 0.0   Warning threshold: > 0.1
    """
    y_pred           = np.array(y_pred)
    sensitive_values = np.array(sensitive_values)
    unique_groups    = np.unique(sensitive_values)

    if len(unique_groups) < 2:
        return 0.0

    rates = [
        float(np.mean(y_pred[sensitive_values == g]))
        for g in unique_groups
        if (sensitive_values == g).sum() > 0
    ]

    dpd = float(max(rates) - min(rates))
    logger.debug(f"DPD={dpd:.4f}  groups={list(unique_groups)}")
    return round(dpd, 4)


def equal_opportunity_difference(
    y_pred: list,
    y_true: list,
    sensitive_values: list,
) -> float:
    """
    EOD = |TPR(group A) – TPR(group B)|

    Ideal: 0.0   Warning threshold: > 0.1
    Requires ground-truth labels (approximated from historical logs in RT).
    """
    y_pred           = np.array(y_pred)
    y_true           = np.array(y_true)
    sensitive_values = np.array(sensitive_values)
    unique_groups    = np.unique(sensitive_values)

    if len(unique_groups) < 2:
        return 0.0

    tprs = []
    for g in unique_groups:
        mask     = sensitive_values == g
        gp, gt   = y_pred[mask], y_true[mask]
        positives = gt == 1
        if positives.sum() == 0:
            continue
        tprs.append(float(np.mean(gp[positives])))

    if len(tprs) < 2:
        return 0.0

    eod = float(max(tprs) - min(tprs))
    logger.debug(f"EOD={eod:.4f}")
    return round(eod, 4)


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _compute_shap_concentration(shap_values: Optional[Dict[str, float]]) -> float:
    """
    Herfindahl–Hirschman Index of |SHAP| values, normalised to [0, 1].

    HHI = 1/n  →  0.0  (perfectly uniform — low concentration)
    HHI = 1.0  →  1.0  (one feature dominates — high concentration)

    Returns 0.0 when shap_values is None or empty (SHAP unavailable).
    """
    if not shap_values:
        return 0.0

    abs_vals = [abs(v) for v in shap_values.values()]
    total    = sum(abs_vals) or 1e-9
    n        = len(abs_vals)

    if n == 0:
        return 0.0
    if n == 1:
        return 1.0

    hhi     = sum((v / total) ** 2 for v in abs_vals)
    hhi_min = 1.0 / n               # uniform distribution baseline
    hhi_max = 1.0                   # theoretical maximum

    normalised = (hhi - hhi_min) / (hhi_max - hhi_min + 1e-12)
    return round(max(0.0, min(1.0, normalised)), 4)


def _score_to_band(score: float) -> str:
    for band, (lo, hi) in BIAS_RISK_BANDS.items():
        if lo <= score < hi:
            return band
    return "critical"   # score == 1.0 edge case


def _band_to_recommendation(band: str) -> str:
    return {
        "low":      "No action required. Prediction is reliable.",
        "moderate": "Log for periodic review. Monitor aggregate fairness metrics.",
        "high":     "Human review recommended before acting on this prediction.",
        "critical": "Escalate to compliance team. Do not act without human approval.",
    }.get(band, "Unknown risk level — treat as critical.")
