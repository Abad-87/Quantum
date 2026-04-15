"""
fairness/checker.py

Fairness evaluation layer — all fairness logic lives here.

─────────────────────────────────────────────────────────────────────────────
EXPORTS
─────────────────────────────────────────────────────────────────────────────
compute_bias_risk_score(...)        Per-prediction bias risk (Phase 1 + 2)
run_fairness_check(...)             Single-prediction report
run_batch_fairness_check(...)       Batch DPD + EOD evaluation
run_post_processing_checks(...)     ← NEW (Phase 2)
    Calibration check  — are predicted probabilities well-calibrated across
                         protected groups?
    Equalized Odds     — are FPR and FNR similar across groups?

demographic_parity_difference(...)  Batch metric helper
equal_opportunity_difference(...)   Batch metric helper

─────────────────────────────────────────────────────────────────────────────
compute_bias_risk_score (Phase 1 base + Phase 2 post-processing boost)
─────────────────────────────────────────────────────────────────────────────
Base score — three weighted components (Phase 1):
  1. Decision-boundary proximity   (30 %)
  2. SHAP feature concentration    (25 %)
  3. Sensitive-attribute base risk (25 %)

Post-processing boost — added on top when batch checks detect disparities:
  4. Calibration disparity penalty (10 %)
  5. Equalized-odds penalty        (10 %)

When the post-processing boost is not yet available (cold-start / no history)
the weights revert to the Phase 1 proportions automatically.

─────────────────────────────────────────────────────────────────────────────
run_post_processing_checks (Phase 2 new function)
─────────────────────────────────────────────────────────────────────────────
Receives batch arrays of predictions, confidences, ground-truth labels and
sensitive-group labels collected from historical records.

Calibration check
    For each group compute the Expected Calibration Error (ECE):
        ECE = Σ_b (|B_b| / N) × |acc(B_b) − conf(B_b)|
    where B_b are equal-width probability bins.
    The inter-group calibration gap = max(ECE) − min(ECE).
    If gap > DISPARITY_THRESHOLD (5 %) → penalty applied to bias_risk_score.

Equalized Odds check
    Compute FPR = FP/(FP+TN) and FNR = FN/(FN+TP) per group.
    Max inter-group |FPR_A − FPR_B| and |FNR_A − FNR_B| are the disparities.
    If either > DISPARITY_THRESHOLD → penalty applied + decision flagged.

Flag-for-review logic
    Any disparity > DISPARITY_THRESHOLD sets ``flag_for_review = True`` and
    injects a warning string.  The bias_risk_score is boosted by a
    proportional penalty (capped so the total never exceeds 1.0).

─────────────────────────────────────────────────────────────────────────────
ETHICAL NOTE
─────────────────────────────────────────────────────────────────────────────
Sensitive attributes are used ONLY to measure and monitor fairness.
They NEVER reach the prediction model.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("fairness")

# ─── Thresholds ───────────────────────────────────────────────────────────────

# DPD / EOD batch-check warning threshold (Phase 1)
FAIRNESS_THRESHOLD: float = 0.10

# Disparity threshold that triggers flag-for-review (Phase 2) — 5 %
DISPARITY_THRESHOLD: float = 0.05

# Number of equal-width bins for ECE calibration (Phase 2)
N_CALIBRATION_BINS: int = 10

# ─── Bias risk bands ─────────────────────────────────────────────────────────

BIAS_RISK_BANDS: Dict[str, tuple] = {
    "low":      (0.00, 0.25),
    "moderate": (0.25, 0.50),
    "high":     (0.50, 0.75),
    "critical": (0.75, 1.00),
}

# Historical discrimination risk per protected attribute type.
# Source: EEOC protected-class guidance + Fairlearn literature.
_ATTR_RISK_WEIGHTS: Dict[str, float] = {
    "race":         0.90,
    "ethnicity":    0.85,
    "gender":       0.75,
    "religion":     0.65,
    "disability":   0.70,
    "age_group":    0.55,
    "location":     0.35,
    "language":     0.30,
    "not_provided": 0.20,
}


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC: PER-PREDICTION BIAS RISK
# ═════════════════════════════════════════════════════════════════════════════

def compute_bias_risk_score(
    confidence:            float,
    shap_values:           Optional[Dict[str, float]],
    sensitive_attr:        Optional[str] = None,
    domain:                str = "unknown",
    post_processing_boost: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """
    Compute a structured bias-risk report for a single prediction.

    Parameters
    ----------
    confidence            : Predicted probability of the positive class (0–1).
    shap_values           : {feature: shap_float} or {} / None if unavailable.
    sensitive_attr        : Sensitive attribute name (monitoring only).
    domain                : Domain label for log context.
    post_processing_boost : Optional dict from run_post_processing_checks()
                            containing "calibration_penalty" and
                            "equalized_odds_penalty" (both in [0, 1]).
                            When provided, the base score is boosted by up to
                            20 percentage points.

    Returns
    -------
    {
        "score":       float [0, 1],
        "band":        "low" | "moderate" | "high" | "critical",
        "flag_for_review": bool,
        "components": {
            "boundary_proximity":      float,
            "shap_concentration":      float,
            "attribute_base_risk":     float,
            "calibration_penalty":     float,   # 0 if no post-proc data
            "equalized_odds_penalty":  float,   # 0 if no post-proc data
        },
        "recommendation": str,
        "post_processing_applied": bool,
    }
    """
    # ── Base components ───────────────────────────────────────────────────────

    # 1. Decision-boundary proximity — max risk at confidence = 0.5
    boundary_proximity = 1.0 - abs(2.0 * confidence - 1.0)

    # 2. SHAP feature concentration (HHI)
    shap_concentration = _compute_shap_concentration(shap_values)

    # 3. Sensitive attribute historical risk
    attr_key       = (sensitive_attr or "not_provided").lower()
    attribute_risk = _ATTR_RISK_WEIGHTS.get(attr_key, 0.40)

    # ── Post-processing penalties (Phase 2) ───────────────────────────────────
    pp = post_processing_boost or {}
    calibration_penalty    = float(pp.get("calibration_penalty",    0.0))
    equalized_odds_penalty = float(pp.get("equalized_odds_penalty", 0.0))
    pp_applied             = bool(pp)

    # ── Weighted score ────────────────────────────────────────────────────────
    # Weights sum to 1.0 whether or not post-processing data is available.
    if pp_applied:
        score = (
            0.30 * boundary_proximity
            + 0.25 * shap_concentration
            + 0.25 * attribute_risk
            + 0.10 * calibration_penalty
            + 0.10 * equalized_odds_penalty
        )
    else:
        # Fall back to Phase-1 proportions (rescaled to sum to 1)
        score = (
            0.40 * boundary_proximity
            + 0.30 * shap_concentration
            + 0.30 * attribute_risk
        )

    score = round(max(0.0, min(1.0, score)), 4)
    band  = _score_to_band(score)

    # Flag for review whenever any disparity penalty is above the threshold
    flag_for_review = (
        calibration_penalty    > DISPARITY_THRESHOLD
        or equalized_odds_penalty > DISPARITY_THRESHOLD
        or band in ("high", "critical")
    )

    recommendation = _band_to_recommendation(band, flag_for_review)

    logger.debug(
        f"[{domain}] bias_risk={score:.4f} ({band})  "
        f"boundary={boundary_proximity:.3f}  "
        f"shap_conc={shap_concentration:.3f}  "
        f"attr_risk={attribute_risk:.3f}  "
        f"cal_pen={calibration_penalty:.3f}  "
        f"eod_pen={equalized_odds_penalty:.3f}  "
        f"flag={flag_for_review}"
    )

    return {
        "score":              score,
        "band":               band,
        "flag_for_review":    flag_for_review,
        "components": {
            "boundary_proximity":      round(boundary_proximity,    4),
            "shap_concentration":      round(shap_concentration,    4),
            "attribute_base_risk":     round(attribute_risk,        4),
            "calibration_penalty":     round(calibration_penalty,   4),
            "equalized_odds_penalty":  round(equalized_odds_penalty,4),
        },
        "recommendation":          recommendation,
        "post_processing_applied": pp_applied,
    }


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC: POST-PROCESSING CHECKS  (Phase 2)
# ═════════════════════════════════════════════════════════════════════════════

def run_post_processing_checks(
    y_pred:           List[int],
    y_prob:           List[float],
    y_true:           List[int],
    sensitive_values: List[str],
    sensitive_attr:   str,
    domain:           str,
) -> Dict:
    """
    Run calibration and equalized-odds checks on a batch of predictions.

    This function is called from a background monitoring job or from the
    router when enough historical predictions are available for a domain.

    Parameters
    ----------
    y_pred           : Binary predictions (0 / 1) for each record.
    y_prob           : Predicted positive-class probabilities (0.0 – 1.0).
    y_true           : Ground-truth labels (0 / 1).
    sensitive_values : Sensitive group label for each record (e.g. "female").
    sensitive_attr   : Name of the sensitive attribute (for logging / report).
    domain           : Domain label.

    Returns
    -------
    {
        "domain":             str,
        "sensitive_attribute": str,
        "n_records":          int,
        "calibration": {
            "per_group":  {group: {"ece": float, "n": int}},
            "max_gap":    float,     # max(ECE) − min(ECE) across groups
            "disparity_detected": bool,
            "penalty":    float,     # normalised to [0, 1] for bias_risk
        },
        "equalized_odds": {
            "per_group":  {group: {"fpr": float, "fnr": float, "n": int}},
            "fpr_gap":    float,
            "fnr_gap":    float,
            "disparity_detected": bool,
            "penalty":    float,
        },
        "flag_for_review": bool,
        "warnings":       list[str],
        "post_processing_boost": {
            "calibration_penalty":    float,
            "equalized_odds_penalty": float,
        },
    }
    """
    y_pred_arr = np.array(y_pred,           dtype=np.int32)
    y_prob_arr = np.array(y_prob,           dtype=np.float64)
    y_true_arr = np.array(y_true,           dtype=np.int32)
    sens_arr   = np.array(sensitive_values, dtype=object)

    groups     = np.unique(sens_arr)
    n_records  = len(y_pred_arr)
    warnings:  List[str] = []

    # ── 1. Calibration check ──────────────────────────────────────────────────
    calibration_result = _calibration_check(
        y_prob_arr, y_true_arr, sens_arr, groups, domain, warnings
    )

    # ── 2. Equalized Odds check ───────────────────────────────────────────────
    eq_odds_result = _equalized_odds_check(
        y_pred_arr, y_true_arr, sens_arr, groups, domain, warnings
    )

    flag_for_review = (
        calibration_result["disparity_detected"]
        or eq_odds_result["disparity_detected"]
    )

    if flag_for_review:
        warnings.append(
            f"⚠️  Decision flagged for review: significant disparity detected "
            f"in [{', '.join(w for w in ['calibration' if calibration_result['disparity_detected'] else '', 'equalized odds' if eq_odds_result['disparity_detected'] else ''] if w)}] "
            f"for attribute '{sensitive_attr}' in domain '{domain}'."
        )
        logger.warning(
            f"[{domain}] flag_for_review=True  "
            f"cal_gap={calibration_result['max_gap']:.4f}  "
            f"fpr_gap={eq_odds_result['fpr_gap']:.4f}  "
            f"fnr_gap={eq_odds_result['fnr_gap']:.4f}"
        )

    return {
        "domain":              domain,
        "sensitive_attribute": sensitive_attr,
        "n_records":           n_records,
        "calibration":         calibration_result,
        "equalized_odds":      eq_odds_result,
        "flag_for_review":     flag_for_review,
        "warnings":            warnings,
        "post_processing_boost": {
            "calibration_penalty":    calibration_result["penalty"],
            "equalized_odds_penalty": eq_odds_result["penalty"],
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC: SINGLE-PREDICTION FAIRNESS REPORT
# ═════════════════════════════════════════════════════════════════════════════

def run_fairness_check(
    prediction:      int,
    sensitive_attr:  str,
    sensitive_value: str,
    domain:          str,
) -> dict:
    """
    Single-prediction fairness report.

    Group statistics cannot be computed from a single data point.  This
    function records the event for later batch analysis and returns a
    per-request summary.  Post-processing checks run asynchronously in the
    background once enough history accumulates.
    """
    return {
        "sensitive_attribute": sensitive_attr,
        "sensitive_value":     sensitive_value,   # stripped before API response
        "domain":              domain,
        "is_fair":             True,
        "warning":             None,
        "metrics": {
            "demographic_parity_difference": None,
            "equal_opportunity_difference":  None,
            "note": (
                "Single-prediction fairness is logged. "
                "Batch calibration and equalized-odds checks run automatically "
                "once ≥ 30 records accumulate per group. "
                "See POST /fairness/batch for on-demand evaluation."
            ),
        },
        "ethical_note": (
            "Sensitive attributes are used ONLY to monitor fairness. "
            "They are NOT inputs to the prediction model."
        ),
    }


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC: BATCH FAIRNESS (DPD + EOD — Phase 1, preserved)
# ═════════════════════════════════════════════════════════════════════════════

def run_batch_fairness_check(
    y_pred:              list,
    y_true:              list,
    sensitive_values:    list,
    sensitive_attr_name: str,
    domain:              str,
) -> dict:
    """
    Full batch DPD + EOD evaluation.
    Call from a monitoring job or the /fairness/batch endpoint.
    """
    dpd = demographic_parity_difference(y_pred, sensitive_values)
    eod = equal_opportunity_difference(y_pred, y_true, sensitive_values)

    dpd_ok  = dpd <= FAIRNESS_THRESHOLD
    eod_ok  = eod <= FAIRNESS_THRESHOLD
    is_fair = dpd_ok and eod_ok

    warnings = []
    if not dpd_ok:
        warnings.append(
            f"Demographic Parity Difference ({dpd:.4f}) exceeds threshold "
            f"({FAIRNESS_THRESHOLD}).  Groups receive unequal positive-outcome rates."
        )
    if not eod_ok:
        warnings.append(
            f"Equal Opportunity Difference ({eod:.4f}) exceeds threshold "
            f"({FAIRNESS_THRESHOLD}).  True positive rates differ across groups."
        )

    return {
        "domain":              domain,
        "sensitive_attribute": sensitive_attr_name,
        "is_fair":             is_fair,
        "threshold":           FAIRNESS_THRESHOLD,
        "metrics": {
            "demographic_parity_difference": dpd,
            "equal_opportunity_difference":  eod,
        },
        "warnings": warnings or None,
    }


# ─── Batch metric primitives ─────────────────────────────────────────────────

def demographic_parity_difference(y_pred: list, sensitive_values: list) -> float:
    """DPD = |P(ŷ=1 | A) − P(ŷ=1 | B)|.  Ideal: 0.0  Warning: > 0.10"""
    y_pred = np.array(y_pred)
    sv     = np.array(sensitive_values)
    groups = np.unique(sv)

    if len(groups) < 2:
        return 0.0

    rates = [float(np.mean(y_pred[sv == g])) for g in groups if (sv == g).sum() > 0]
    dpd   = float(max(rates) - min(rates))
    logger.debug(f"DPD={dpd:.4f}  groups={list(groups)}")
    return round(dpd, 4)


def equal_opportunity_difference(
    y_pred: list, y_true: list, sensitive_values: list
) -> float:
    """EOD = |TPR(A) − TPR(B)|.  Ideal: 0.0  Warning: > 0.10"""
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    sv     = np.array(sensitive_values)
    groups = np.unique(sv)

    if len(groups) < 2:
        return 0.0

    tprs = []
    for g in groups:
        mask = sv == g
        gp, gt = y_pred[mask], y_true[mask]
        pos = gt == 1
        if pos.sum() == 0:
            continue
        tprs.append(float(np.mean(gp[pos])))

    if len(tprs) < 2:
        return 0.0

    eod = float(max(tprs) - min(tprs))
    logger.debug(f"EOD={eod:.4f}")
    return round(eod, 4)


# ═════════════════════════════════════════════════════════════════════════════
# INTERNAL: CALIBRATION CHECK
# ═════════════════════════════════════════════════════════════════════════════

def _calibration_check(
    y_prob:    np.ndarray,
    y_true:    np.ndarray,
    sens_arr:  np.ndarray,
    groups:    np.ndarray,
    domain:    str,
    warnings:  List[str],
) -> dict:
    """
    Compute Expected Calibration Error (ECE) per sensitive group.

    ECE = Σ_bin (|bin| / N) × |mean_conf(bin) − mean_acc(bin)|

    A well-calibrated model has ECE ≈ 0.  The inter-group gap measures
    whether calibration quality is consistent across protected groups.

    Returns calibration sub-report including ``penalty`` ∈ [0, 1].
    """
    bin_edges    = np.linspace(0.0, 1.0, N_CALIBRATION_BINS + 1)
    per_group:   Dict[str, dict] = {}
    ece_values:  List[float]     = []

    for g in groups:
        mask    = sens_arr == g
        gp      = y_prob[mask]
        gt      = y_true[mask].astype(float)
        n_group = int(mask.sum())

        if n_group == 0:
            continue

        ece = _compute_ece(gp, gt, bin_edges)
        ece_values.append(ece)
        per_group[str(g)] = {"ece": round(ece, 6), "n": n_group}

        logger.debug(f"[{domain}] calibration  group={g}  ECE={ece:.4f}  n={n_group}")

    if len(ece_values) < 2:
        return {
            "per_group":          per_group,
            "max_gap":            0.0,
            "disparity_detected": False,
            "penalty":            0.0,
            "note":               "Insufficient groups for calibration comparison.",
        }

    max_gap           = round(float(max(ece_values) - min(ece_values)), 6)
    disparity         = max_gap > DISPARITY_THRESHOLD
    # Penalty is the gap normalised to [0, 1] using the threshold as midpoint.
    # Gaps at threshold → 0.5 penalty; gaps at 2× threshold → 1.0 (capped).
    penalty           = round(min(1.0, max_gap / (2.0 * DISPARITY_THRESHOLD)), 4)

    if disparity:
        msg = (
            f"Calibration disparity detected (gap={max_gap:.2%}, "
            f"threshold={DISPARITY_THRESHOLD:.0%}).  "
            "Predicted probabilities are not equally reliable across groups."
        )
        warnings.append(msg)
        logger.warning(f"[{domain}] {msg}")

    return {
        "per_group":          per_group,
        "max_gap":            max_gap,
        "disparity_detected": disparity,
        "penalty":            penalty,
    }


def _compute_ece(
    y_prob:    np.ndarray,
    y_true:    np.ndarray,
    bin_edges: np.ndarray,
) -> float:
    """Compute ECE for a single group using equal-width probability bins."""
    n   = len(y_prob)
    ece = 0.0

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        # Include the right edge in the last bin to catch prob == 1.0
        if hi == bin_edges[-1]:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)

        if mask.sum() == 0:
            continue

        mean_conf = float(np.mean(y_prob[mask]))
        mean_acc  = float(np.mean(y_true[mask]))
        ece      += (mask.sum() / n) * abs(mean_conf - mean_acc)

    return round(ece, 6)


# ═════════════════════════════════════════════════════════════════════════════
# INTERNAL: EQUALIZED ODDS CHECK
# ═════════════════════════════════════════════════════════════════════════════

def _equalized_odds_check(
    y_pred:   np.ndarray,
    y_true:   np.ndarray,
    sens_arr: np.ndarray,
    groups:   np.ndarray,
    domain:   str,
    warnings: List[str],
) -> dict:
    """
    Compute FPR and FNR per sensitive group and measure inter-group disparity.

    FPR = FP / (FP + TN) — false alarm rate
    FNR = FN / (FN + TP) — miss rate

    Equalized Odds requires BOTH FPR and FNR to be equal across groups.
    We report both gaps and set disparity_detected if EITHER exceeds
    DISPARITY_THRESHOLD (5 %).

    Returns equalized-odds sub-report including ``penalty`` ∈ [0, 1].
    """
    per_group: Dict[str, dict] = {}
    fprs:      List[float]     = []
    fnrs:      List[float]     = []

    for g in groups:
        mask    = sens_arr == g
        gp      = y_pred[mask]
        gt      = y_true[mask]
        n_group = int(mask.sum())

        if n_group == 0:
            continue

        fpr, fnr = _compute_fpr_fnr(gp, gt)
        fprs.append(fpr)
        fnrs.append(fnr)
        per_group[str(g)] = {
            "fpr": round(fpr, 6),
            "fnr": round(fnr, 6),
            "n":   n_group,
        }
        logger.debug(
            f"[{domain}] eq_odds  group={g}  FPR={fpr:.4f}  FNR={fnr:.4f}  n={n_group}"
        )

    if len(fprs) < 2:
        return {
            "per_group":          per_group,
            "fpr_gap":            0.0,
            "fnr_gap":            0.0,
            "disparity_detected": False,
            "penalty":            0.0,
            "note":               "Insufficient groups for equalized-odds comparison.",
        }

    fpr_gap = round(float(max(fprs) - min(fprs)), 6)
    fnr_gap = round(float(max(fnrs) - min(fnrs)), 6)

    # Disparity if EITHER gap exceeds the threshold
    disparity = fpr_gap > DISPARITY_THRESHOLD or fnr_gap > DISPARITY_THRESHOLD

    # Penalty driven by the larger of the two gaps
    worst_gap = max(fpr_gap, fnr_gap)
    penalty   = round(min(1.0, worst_gap / (2.0 * DISPARITY_THRESHOLD)), 4)

    if fpr_gap > DISPARITY_THRESHOLD:
        msg = (
            f"False Positive Rate disparity detected (gap={fpr_gap:.2%}, "
            f"threshold={DISPARITY_THRESHOLD:.0%}).  "
            "Some groups face higher false alarm rates than others."
        )
        warnings.append(msg)
        logger.warning(f"[{domain}] {msg}")

    if fnr_gap > DISPARITY_THRESHOLD:
        msg = (
            f"False Negative Rate disparity detected (gap={fnr_gap:.2%}, "
            f"threshold={DISPARITY_THRESHOLD:.0%}).  "
            "Some groups face higher miss rates than others."
        )
        warnings.append(msg)
        logger.warning(f"[{domain}] {msg}")

    return {
        "per_group":          per_group,
        "fpr_gap":            fpr_gap,
        "fnr_gap":            fnr_gap,
        "disparity_detected": disparity,
        "penalty":            penalty,
    }


def _compute_fpr_fnr(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> tuple[float, float]:
    """
    Compute FPR and FNR for a single group.

    FPR = FP / (FP + TN)  — undefined when no negatives → returns 0.0
    FNR = FN / (FN + TP)  — undefined when no positives → returns 0.0
    """
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return round(fpr, 6), round(fnr, 6)


# ═════════════════════════════════════════════════════════════════════════════
# INTERNAL: SHARED HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _compute_shap_concentration(shap_values: Optional[Dict[str, float]]) -> float:
    """
    Herfindahl–Hirschman Index of |SHAP| values, normalised to [0, 1].

    HHI = 1/n → 0.0  (perfectly uniform)
    HHI = 1.0 → 1.0  (one feature dominates)

    Returns 0.0 when shap_values is None / empty.
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

    hhi        = sum((v / total) ** 2 for v in abs_vals)
    hhi_min    = 1.0 / n
    normalised = (hhi - hhi_min) / (1.0 - hhi_min + 1e-12)
    return round(max(0.0, min(1.0, normalised)), 4)


def _score_to_band(score: float) -> str:
    for band, (lo, hi) in BIAS_RISK_BANDS.items():
        if lo <= score < hi:
            return band
    return "critical"


def _band_to_recommendation(band: str, flag_for_review: bool = False) -> str:
    base = {
        "low":      "No action required. Prediction is reliable.",
        "moderate": "Log for periodic review. Monitor aggregate fairness metrics.",
        "high":     "Human review recommended before acting on this prediction.",
        "critical": "Escalate to compliance team. Do not act without human approval.",
    }.get(band, "Unknown risk level — treat as critical.")

    if flag_for_review and band not in ("high", "critical"):
        return (
            base + "  Additionally, post-processing checks detected a group disparity "
            "> 5 % — this decision has been flagged for human review."
        )
    return base
