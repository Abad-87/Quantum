"""
social/predictor.py

Prediction logic for Social Media Recommendation domain.

This model recommends content categories to show a user
based on their behavioral features — NOT their demographics.

Model features (no sensitive attributes):
- avg_session_minutes  : average session length
- posts_per_day        : content creation frequency
- topics_interacted    : encoded list of topics user engaged with
- like_rate            : ratio of liked to seen content (0.0–1.0)
- share_rate           : ratio of shared to liked content (0.0–1.0)
- comment_rate         : ratio of commented to seen content (0.0–1.0)
- account_age_days     : how long the account has existed

Output: recommended content category ID (0–N) + label + explanation
"""

import numpy as np
import logging

logger = logging.getLogger("social.predictor")

FEATURE_NAMES = [
    "avg_session_minutes",
    "posts_per_day",
    "topics_interacted",
    "like_rate",
    "share_rate",
    "comment_rate",
    "account_age_days",
]

# Map numeric category ID → human-readable label
CONTENT_CATEGORIES = {
    0: "Technology & Science",
    1: "Entertainment & Pop Culture",
    2: "Sports & Fitness",
    3: "News & Politics",
    4: "Arts & Creativity",
    5: "Education & Learning",
    6: "Business & Finance",
    7: "Lifestyle & Wellness",
}


def predict(model, features: dict) -> tuple[int, str, float, str]:
    """
    Returns (category_id, category_label, confidence, explanation).
    """
    input_row = [[
        features["avg_session_minutes"],
        features["posts_per_day"],
        features["topics_interacted"],
        features["like_rate"],
        features["share_rate"],
        features["comment_rate"],
        features["account_age_days"],
    ]]

    category_id = int(model.predict(input_row)[0])
    category_label = CONTENT_CATEGORIES.get(category_id, f"Category {category_id}")

    confidence = 0.5
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_row)[0]
        confidence = round(float(np.max(proba)), 3)

    explanation = _explain(model, features, category_id, category_label, input_row)
    return category_id, category_label, confidence, explanation


def _explain(model, features: dict, category_id: int, category_label: str, input_row: list) -> str:
    try:
        import shap
        base_model = model
        if hasattr(model, "steps"):
            base_model = model.steps[-1][1]

        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(input_row)

        # For multiclass, shap_values is a list of arrays, one per class
        if isinstance(shap_values, list) and len(shap_values) > category_id:
            values = shap_values[category_id][0]
        elif isinstance(shap_values, np.ndarray):
            values = shap_values[0]
        else:
            raise ValueError("Unexpected SHAP output format")

        shap_pairs = sorted(
            zip(FEATURE_NAMES, values),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        top_feature, _ = shap_pairs[0]
        pretty = top_feature.replace("_", " ")
        val = features.get(top_feature, "N/A")

        return (
            f"Recommended '{category_label}' based on your engagement patterns, "
            f"primarily driven by {pretty} ({val})."
        )

    except Exception as e:
        logger.debug(f"SHAP unavailable ({e}). Using rule-based explanation.")
        return _rule_based_explanation(features, category_label)


def _rule_based_explanation(features: dict, category_label: str) -> str:
    like_rate    = features.get("like_rate", 0)
    share_rate   = features.get("share_rate", 0)
    session_mins = features.get("avg_session_minutes", 0)
    posts        = features.get("posts_per_day", 0)

    signals = []
    if like_rate > 0.6:
        signals.append("high content engagement")
    if share_rate > 0.3:
        signals.append("active content sharing")
    if session_mins > 30:
        signals.append(f"long session activity ({session_mins:.0f} min avg)")
    if posts > 2:
        signals.append("frequent posting")

    signals_str = ", ".join(signals) if signals else "your recent activity patterns"
    return (
        f"Recommended '{category_label}' based on {signals_str}. "
        "This recommendation is based on behavior only, not personal attributes."
    )
