"""
create_dummy_models.py

Creates placeholder .pkl model files so you can test the API
before the AI team provides real trained models.

These are real scikit-learn Random Forest models trained on random data —
they will produce actual predictions (though not meaningful ones).

Run:  python create_dummy_models.py
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

Path("models").mkdir(exist_ok=True)

print("Creating dummy models for testing...\n")

# ─── 1. Hiring Model ──────────────────────────────────────────────────────────
# Features: years_experience, education_level, technical_score,
#           communication_score, num_past_jobs, certifications
print("Creating hiring_model.pkl ...")
np.random.seed(42)
X_hiring = np.random.rand(500, 6) * [20, 3, 100, 100, 10, 5]
y_hiring  = (X_hiring[:, 2] + X_hiring[:, 3] > 100).astype(int)  # hired if scores > 100
hiring_model = RandomForestClassifier(n_estimators=50, random_state=42)
hiring_model.fit(X_hiring, y_hiring)
joblib.dump(hiring_model, "models/hiring_model.pkl")
print("  ✅ models/hiring_model.pkl created")

# ─── 2. Loan Model ───────────────────────────────────────────────────────────
# Features: credit_score, annual_income, loan_amount, loan_term_months,
#           employment_years, existing_debt, num_credit_lines
print("Creating loan_model.pkl ...")
X_loan = np.column_stack([
    np.random.randint(300, 850, 500),        # credit_score
    np.random.randint(20000, 200000, 500),   # annual_income
    np.random.randint(1000, 100000, 500),    # loan_amount
    np.random.choice([12, 24, 36, 60], 500), # loan_term_months
    np.random.randint(0, 20, 500),           # employment_years
    np.random.randint(0, 50000, 500),        # existing_debt
    np.random.randint(0, 10, 500),           # num_credit_lines
])
# Approved if credit_score > 620 and debt-to-income < 0.5
dti = X_loan[:, 5] / (X_loan[:, 1] + 1)
y_loan = ((X_loan[:, 0] > 620) & (dti < 0.5)).astype(int)
loan_model = RandomForestClassifier(n_estimators=50, random_state=42)
loan_model.fit(X_loan, y_loan)
joblib.dump(loan_model, "models/loan_model.pkl")
print("  ✅ models/loan_model.pkl created")

# ─── 3. Social Recommendation Model ──────────────────────────────────────────
# Features: avg_session_minutes, posts_per_day, topics_interacted,
#           like_rate, share_rate, comment_rate, account_age_days
# Output: category 0–7
print("Creating social_model.pkl ...")
X_social = np.column_stack([
    np.random.rand(500) * 120,        # avg_session_minutes
    np.random.rand(500) * 10,         # posts_per_day
    np.random.randint(1, 30, 500),    # topics_interacted
    np.random.rand(500),              # like_rate
    np.random.rand(500) * 0.5,        # share_rate
    np.random.rand(500) * 0.3,        # comment_rate
    np.random.randint(1, 2000, 500),  # account_age_days
])
y_social = np.random.randint(0, 8, 500)  # 8 content categories
social_model = RandomForestClassifier(n_estimators=50, random_state=42)
social_model.fit(X_social, y_social)
joblib.dump(social_model, "models/social_model.pkl")
print("  ✅ models/social_model.pkl created")

print("\n✅ All dummy models created in /models directory.")
print("   Run: uvicorn main:app --reload  to start the server.")
print("   Visit: http://localhost:8000/docs  to test the APIs.")
