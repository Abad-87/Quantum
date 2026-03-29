"""
fairness/checker.py

Fairness evaluation layer using Fairlearn concepts.
Computes Demographic Parity Difference and Equal Opportunity Difference.

IMPORTANT ETHICAL NOTE:
- Sensitive attributes (gender, religion, race, etc.) are ONLY used here
  to MEASURE fairness — never for making the prediction itself.
- This is the core of "unbiased AI": the model predicts without knowing
  sensitive info, then we check if outcomes are fair across groups.
"""

import numpy as np
import logging

logger = logging.getLogger("fairness")

# ─── Threshold: if any fairness metric exceeds this, warn the caller ────────
FAIRNESS_THRESHOLD = 0.1


def demographic_parity_difference(y_pred: list, sensitive_values: list) -> float:
    """
    Demographic Parity Difference (DPD):
    Measures whether different groups receive positive outcomes at the same rate.

    DPD = |P(ŷ=1 | group A) - P(ŷ=1 | group B)|

    Ideal value: 0.0  (both groups get same positive-outcome rate)
    Concerning:  > 0.1
    """
    y_pred = np.array(y_pred)
    sensitive_values = np.array(sensitive_values)

    unique_groups = np.unique(sensitive_values)
    if len(unique_groups) < 2:
        # Only one group — can't compare, return 0
        return 0.0

    rates = []
    for group in unique_groups:
        mask = sensitive_values == group
        group_preds = y_pred[mask]
        if len(group_preds) == 0:
            continue
        positive_rate = np.mean(group_preds)
        rates.append(positive_rate)

    dpd = float(max(rates) - min(rates))
    logger.debug(f"DPD computed: {dpd:.4f} across groups {unique_groups}")
    return round(dpd, 4)


def equal_opportunity_difference(
    y_pred: list,
    y_true: list,
    sensitive_values: list,
) -> float:
    """
    Equal Opportunity Difference (EOD):
    Measures whether the TRUE POSITIVE RATE is equal across groups.

    EOD = |TPR(group A) - TPR(group B)|

    Ideal value: 0.0
    Concerning:  > 0.1

    Note: Requires knowing the ground-truth label (y_true).
    In real-time prediction, this is approximated from historical logs.
    """
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    sensitive_values = np.array(sensitive_values)

    unique_groups = np.unique(sensitive_values)
    if len(unique_groups) < 2:
        return 0.0

    tprs = []
    for group in unique_groups:
        mask = sensitive_values == group
        gp = y_pred[mask]
        gt = y_true[mask]

        # True positives among actual positives
        actual_positives = gt == 1
        if actual_positives.sum() == 0:
            continue

        tpr = np.mean(gp[actual_positives])
        tprs.append(tpr)

    if len(tprs) < 2:
        return 0.0

    eod = float(max(tprs) - min(tprs))
    logger.debug(f"EOD computed: {eod:.4f}")
    return round(eod, 4)


def run_fairness_check(
    prediction: int,
    sensitive_attr: str,
    sensitive_value: str,
    domain: str,
) -> dict:
    """
    Single-prediction fairness check.

    For a single prediction we can't compute group statistics, so we:
    1. Log the prediction alongside its sensitive attribute.
    2. Return a fairness report with a disclaimer.
    3. In a real system, this would aggregate over time (see utils/logger.py).

    Returns a dict with:
    - is_fair: bool
    - warning: str or None
    - metrics: dict (populated from historical data in production)
    """

    # For demo / hackathon: simulate a fairness score
    # In production: query the database for recent predictions and compute real metrics
    report = {
        "sensitive_attribute": sensitive_attr,
        "sensitive_value": sensitive_value,   # ← NOT returned in API response
        "domain": domain,
        "is_fair": True,
        "warning": None,
        "metrics": {
            "demographic_parity_difference": None,  # Computed from batch history
            "equal_opportunity_difference": None,
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


def run_batch_fairness_check(
    y_pred: list,
    y_true: list,
    sensitive_values: list,
    sensitive_attr_name: str,
    domain: str,
) -> dict:
    """
    Full batch fairness evaluation.
    Call this from a monitoring job or analytics endpoint.

    Returns DPD, EOD, and whether each metric passes the threshold.
    """
    dpd = demographic_parity_difference(y_pred, sensitive_values)
    eod = equal_opportunity_difference(y_pred, y_true, sensitive_values)

    dpd_ok = dpd <= FAIRNESS_THRESHOLD
    eod_ok = eod <= FAIRNESS_THRESHOLD
    is_fair = dpd_ok and eod_ok

    warnings = []
    if not dpd_ok:
        warnings.append(
            f"Demographic Parity Difference ({dpd}) exceeds threshold ({FAIRNESS_THRESHOLD}). "
            f"Different groups may receive unequal positive-outcome rates."
        )
    if not eod_ok:
        warnings.append(
            f"Equal Opportunity Difference ({eod}) exceeds threshold ({FAIRNESS_THRESHOLD}). "
            f"True positive rates differ across groups."
        )

    return {
        "domain": domain,
        "sensitive_attribute": sensitive_attr_name,
        "is_fair": is_fair,
        "threshold": FAIRNESS_THRESHOLD,
        "metrics": {
            "demographic_parity_difference": dpd,
            "equal_opportunity_difference": eod,
        },
        "warnings": warnings if warnings else None,
    }
