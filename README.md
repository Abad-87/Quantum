# ⚛️ Quantum — Unbiased AI Decision Platform

> A fairness-aware, explainable AI backend for Job Hiring, Loan Approval, and Social Media Recommendation. Built with FastAPI and designed for hackathon demonstration.

---

## 🗂️ Project Structure

```
quantum/
│
├── main.py                    ← FastAPI app entry point
│
├── hiring/                    ← Job Hiring domain
│   ├── __init__.py
│   ├── model_loader.py        ← Loads hiring_model.pkl
│   ├── predictor.py           ← Prediction + SHAP explanation
│   └── router.py              ← POST /hiring/predict endpoint
│
├── loan/                      ← Banking & Loan Approval domain
│   ├── __init__.py
│   ├── model_loader.py        ← Loads loan_model.pkl
│   ├── predictor.py           ← Prediction + SHAP explanation
│   └── router.py              ← POST /loan/predict endpoint
│
├── social/                    ← Social Media Recommendation domain
│   ├── __init__.py
│   ├── model_loader.py        ← Loads social_model.pkl
│   ├── predictor.py           ← Prediction + SHAP explanation
│   └── router.py              ← POST /social/recommend endpoint
│
├── fairness/                  ← Fairness evaluation layer
│   ├── __init__.py
│   └── checker.py             ← DPD, EOD, fairness reports
│
├── utils/                     ← Shared utilities
│   ├── __init__.py
│   ├── logger.py              ← Structured prediction logging
│   └── database.py            ← MongoDB / JSON persistence
│
├── models/                    ← Place .pkl files here
│   └── README.md
│
├── tests/
│   └── test_api.py            ← Full integration test suite
│
├── create_dummy_models.py     ← Generate test .pkl files
├── sample_requests.json       ← Frontend integration reference
├── requirements.txt
├── render.yaml                ← Render cloud deployment config
├── .env.example               ← Environment variable template
└── .gitignore
```

---

## ⚡ Quick Start (Local)

### 1. Clone & Install

```bash
git clone https://github.com/your-team/quantum.git
cd quantum

python -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Create dummy models (if AI team hasn't sent .pkl files yet)

```bash
python create_dummy_models.py
```

This creates `models/hiring_model.pkl`, `models/loan_model.pkl`, `models/social_model.pkl`.

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env — set MONGO_URI if you have MongoDB Atlas, or leave blank for JSON fallback
```

### 4. Start the server

```bash
uvicorn main:app --reload
```

### 5. Open the interactive API docs

```
http://localhost:8000/docs
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/hiring/predict` | Job hiring prediction |
| `POST` | `/loan/predict` | Loan approval prediction |
| `POST` | `/social/recommend` | Content recommendation |
| `GET` | `/` | Platform info |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Swagger UI |

---

## 📡 Sample API Calls

### Job Hiring

```bash
curl -X POST http://localhost:8000/hiring/predict \
  -H "Content-Type: application/json" \
  -d '{
    "years_experience": 5,
    "education_level": 2,
    "technical_score": 82,
    "communication_score": 75,
    "num_past_jobs": 3,
    "certifications": 2,
    "gender": "female"
  }'
```

**Response:**
```json
{
  "prediction": 1,
  "prediction_label": "Hired",
  "confidence": 0.87,
  "explanation": "Hired — candidate shows strong technical score (82/100), 5 years of experience.",
  "fairness": {
    "sensitive_attribute": "gender",
    "is_fair": true,
    "warning": null,
    "ethical_note": "Sensitive attributes are used ONLY to monitor fairness."
  },
  "message": "Prediction complete."
}
```

### Loan Approval

```bash
curl -X POST http://localhost:8000/loan/predict \
  -H "Content-Type: application/json" \
  -d '{
    "credit_score": 720,
    "annual_income": 75000,
    "loan_amount": 25000,
    "loan_term_months": 36,
    "employment_years": 4,
    "existing_debt": 8000,
    "num_credit_lines": 3,
    "ethnicity": "hispanic"
  }'
```

### Social Recommendation

```bash
curl -X POST http://localhost:8000/social/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "avg_session_minutes": 45,
    "posts_per_day": 3,
    "topics_interacted": 12,
    "like_rate": 0.65,
    "share_rate": 0.2,
    "comment_rate": 0.1,
    "account_age_days": 365,
    "age_group": "25-34"
  }'
```

---

## ⚖️ How Fairness Works

```
                    ┌─────────────────────┐
User Request ──────►│  FastAPI Endpoint   │
                    └──────────┬──────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                 │
              ▼                                 ▼
   ┌─────────────────────┐         ┌────────────────────────┐
   │   Prediction Model  │         │   Fairness Checker     │
   │                     │         │                        │
   │  Input:             │         │  Input:                │
   │  - Objective        │         │  - Sensitive attrs     │
   │    features only    │         │    (gender, race, etc) │
   │                     │         │                        │
   │  Output:            │         │  Computes:             │
   │  - Prediction       │         │  - Demographic Parity  │
   │  - Confidence       │         │  - Equal Opportunity   │
   │  - Explanation      │         │  - Warning if > 0.1    │
   └─────────────────────┘         └────────────────────────┘
              │                                 │
              └────────────────┬────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Unified Response  │
                    │  + Audit Log        │
                    └─────────────────────┘
```

### Key Principle: Sensitive attributes are **separated** from prediction features

| Attribute | Used for Prediction? | Used for Fairness Check? |
|-----------|---------------------|--------------------------|
| credit_score | ✅ Yes | ❌ No |
| gender | ❌ No | ✅ Yes (monitoring only) |
| religion | ❌ No | ✅ Yes (monitoring only) |
| ethnicity | ❌ No | ✅ Yes (monitoring only) |

### Fairness Metrics

| Metric | Formula | Ideal | Warning Threshold |
|--------|---------|-------|-------------------|
| Demographic Parity Difference | \|P(ŷ=1\|A) - P(ŷ=1\|B)\| | 0.0 | > 0.1 |
| Equal Opportunity Difference | \|TPR(A) - TPR(B)\| | 0.0 | > 0.1 |

---

## 🧠 Explainability

Every prediction includes a human-readable explanation:

- **With SHAP** (if shap is installed): Uses tree explainer to identify the top contributing feature
- **Without SHAP**: Falls back to rule-based explanation using feature values

Examples:
```
"Hired — primarily driven by high technical score (82/100)."
"Loan rejected — low credit score (480, minimum 600); high debt-to-income ratio (72%)."
"Recommended 'Technology & Science' based on high content engagement (like_rate: 0.65)."
```

---

## 🗄️ Database

| Mode | When | Storage |
|------|------|---------|
| MongoDB (Atlas) | `MONGO_URI` is set | Cloud database — persistent, scalable |
| JSON Fallback | No `MONGO_URI` | `predictions.json` — local file |

Every prediction stores:
- Input features (no sensitive data)
- Prediction result and label
- Confidence score
- Explanation text
- Fairness report
- Timestamp (UTC)

---

## 🧪 Running Tests

```bash
# Create dummy models first
python create_dummy_models.py

# Run all tests
pytest tests/test_api.py -v
```

---

## ☁️ Deploying to Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repo
4. Render detects `render.yaml` automatically
5. In **Environment** tab, add:
   - `MONGO_URI` = your MongoDB Atlas connection string (optional)
6. Click **Deploy**

Your API will be live at: `https://quantum-ai-backend.onrender.com`

> **Note:** Free tier spins down after 15 minutes of inactivity. First request after sleep takes ~30 seconds. Upgrade to Starter ($7/mo) for always-on.

---

## 🔗 Frontend (React) Integration

```javascript
// In your React component:
import axios from 'axios';

const API_BASE = 'https://quantum-ai-backend.onrender.com';

// Hiring prediction
const predictHiring = async (formData) => {
  const response = await axios.post(`${API_BASE}/hiring/predict`, {
    years_experience: formData.experience,
    education_level: formData.education,
    technical_score: formData.techScore,
    communication_score: formData.commScore,
    num_past_jobs: formData.pastJobs,
    certifications: formData.certs,
    gender: formData.gender,    // Optional — for fairness monitoring
  });

  const { prediction_label, confidence, explanation, fairness } = response.data;
  // Display to user...
};
```

---

## 🔐 Security & Ethics

1. **Sensitive data isolation**: Gender, religion, ethnicity never reach the prediction model
2. **Privacy in responses**: `sensitive_value` is stripped before sending response to client
3. **Input validation**: All inputs validated by Pydantic — type, range, required fields
4. **Audit trail**: Every prediction logged for accountability
5. **Explainability**: Every decision explained — no black box outputs

---

## 🎤 Hackathon Demo Tips

### Demonstrate Fairness Live
1. Submit two identical applications — only change `gender`
2. Show predictions are identical → **model is fair**
3. Change the prediction features to show decisions are based on merit
4. Point at the `fairness.ethical_note` in the response

### Show Explainability
1. Submit a loan application with very low credit score
2. Show the explanation: "Rejected due to low credit score (400)"
3. Update the credit score to 750, resubmit
4. Show: "Approved — good credit score (750)"

### Batch Fairness Chart (Bonus)
Run 20 requests with different sensitive attributes → plot DPD values over time using `/docs` or Postman Collection

---

## 🌟 Potential Improvements (Post-Hackathon)

- [ ] `/metrics` endpoint returning aggregate fairness stats dashboard
- [ ] Integrate actual Fairlearn `MetricFrame` for richer group analysis
- [ ] Add JWT authentication for API security
- [ ] Async model inference using background workers
- [ ] WebSocket endpoint for real-time fairness monitoring
- [ ] Bias mitigation: Exponentiated Gradient or Threshold Optimizer from Fairlearn

---

## 👥 Team

| Role | Responsibility |
|------|---------------|
| AI Team | Train & export `.pkl` models |
| Backend Team | This repo — FastAPI integration |
| Frontend Team | React UI consuming the APIs |

---

*Built with FastAPI · scikit-learn · SHAP · Fairlearn · MongoDB*
