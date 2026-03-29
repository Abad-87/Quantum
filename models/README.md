# /models — Pre-trained Model Files

Place your `.pkl` model files here before starting the server.

| File | Domain | Expected Input |
|------|--------|---------------|
| `hiring_model.pkl` | Job Hiring | years_experience, education_level, technical_score, communication_score, num_past_jobs, certifications |
| `loan_model.pkl` | Loan Approval | credit_score, annual_income, loan_amount, loan_term_months, employment_years, existing_debt, num_credit_lines |
| `social_model.pkl` | Social Recommendation | avg_session_minutes, posts_per_day, topics_interacted, like_rate, share_rate, comment_rate, account_age_days |

## Requirements for model files

- Serialized with **joblib** (`joblib.dump(model, "model.pkl")`)
- Must implement `.predict(X)` where X is a 2D array/list
- Should implement `.predict_proba(X)` for confidence scores (optional but recommended)
- Compatible with **scikit-learn 1.4.x**

## Creating a dummy model (for testing)

If you don't have real models yet, run this script to generate placeholder models:

```bash
python create_dummy_models.py
```

This creates random-forest classifiers that respond to the correct feature shapes,
so you can test all API endpoints end-to-end before integrating real AI models.
