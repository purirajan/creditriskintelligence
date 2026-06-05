# CreditRisk Intelligence — AI-Powered Credit Risk Infrastructure for Fintechs


---

## 🎯 What This Is

A modular, production-grade credit risk platform designed for fintechs — BNPL providers, neobanks,
and embedded finance companies — that need enterprise-grade risk infrastructure without the enterprise overhead.

**Three core pillars:**

| Pillar | What it does |
|---|---|
| 🔢 **Risk Scoring API** | Basel-aligned PD/LGD/EAD models exposed as RESTful APIs |
| 🔍 **Model Explainability** | SHAP-based fair lending compliance toolkit |
| 🏛️ **Governance Engine** | Automated model monitoring, drift detection, audit trails |

---

## 📁 Repository Structure

```
creditrisk-intelligence/
│
├── src/
│   ├── models/              # PD, LGD, EAD model classes
│   │   ├── pd_model.py      # Probability of Default (logistic regression, XGBoost)
│   │   ├── lgd_model.py     # Loss Given Default (beta regression)
│   │   ├── ead_model.py     # Exposure at Default (CCF models)
│   │   └── base_model.py    # Abstract base with governance hooks
│   │
│   ├── api/                 # FastAPI endpoints
│   │   ├── main.py          # App entrypoint
│   │   ├── routes/
│   │   │   ├── score.py     # POST /score — real-time risk scoring
│   │   │   ├── explain.py   # POST /explain — SHAP explanations
│   │   │   └── monitor.py   # GET /monitor — model health metrics
│   │   └── schemas.py       # Pydantic request/response schemas
│   │
│   ├── governance/          # Model Risk Management (MRM) framework
│   │   ├── model_card.py    # Auto-generate model documentation
│   │   ├── audit_trail.py   # Immutable logging for regulatory review
│   │   ├── thresholds.py    # Monitoring KPI definitions
│   │   └── validator.py     # Pre-deployment validation checks
│   │
│   ├── explainability/      # Fair lending & model transparency
│   │   ├── shap_engine.py   # SHAP values + force plots
│   │   ├── disparate_impact.py  # Fair lending ratio analysis
│   │   ├── feature_report.py    # Feature importance narratives
│   │   └── ecoa_compliance.py   # ECOA/Reg B adverse action reasons
│   │
│   └── monitoring/          # Ongoing model health
│       ├── drift_detector.py    # PSI / KS-based drift detection
│       ├── performance_tracker.py  # Gini, AUC, KS over time
│       └── alert_engine.py      # Threshold breach notifications
│
├── dashboard/               # React + Recharts monitoring dashboard
│   └── src/App.jsx
│
├── notebooks/               # Exploratory analysis & model dev
│   ├── 01_eda.ipynb
│   ├── 02_pd_model_dev.ipynb
│   └── 03_model_validation.ipynb
│
├── docs/
│   ├── model_governance_framework.md
│   ├── api_reference.md
│   ├── fair_lending_guide.md
│   └── regulatory_mapping.md   # CECL / Basel / CCAR alignment
│
├── tests/
│   ├── test_pd_model.py
│   ├── test_api.py
│   └── test_governance.py
│
├── configs/
│   ├── model_config.yaml    # Model hyperparameters
│   └── monitoring_config.yaml  # Alert thresholds
│
├── scripts/
│   ├── train_models.py      # End-to-end training pipeline
│   └── generate_model_card.py  # Auto-documentation
│
├── .github/
│   └── workflows/
│       ├── ci.yml           # Tests + linting on PR
│       └── model_monitor.yml  # Scheduled monitoring runs
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🚀 Quick Start

```bash
# Clone and install
git clone https://github.com/purirajan/creditrisk-intelligence
cd creditrisk-intelligence
pip install -r requirements.txt

# Start the API
uvicorn src.api.main:app --reload

# Score a borrower
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"fico": 680, "dti": 0.35, "utilization": 0.42, "months_on_book": 18}'

# Get SHAP explanation
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{"application_id": "APP-001", "fico": 680, "dti": 0.35}'
```

---

## 🏦 Regulatory Coverage

| Framework | Coverage |
|---|---|
| **CECL** | Lifetime PD/LGD curves, vintage analysis |
| **Basel II/III** | IRB-compliant PD/LGD/EAD, RWA calculation |
| **CCAR** | Stress scenario simulation (baseline, adverse, severely adverse) |
| **ECOA / Reg B** | Adverse action reason codes, disparate impact testing |
| **SR 11-7** | Model development, validation, and governance documentation |

---

## 🔑 Key Differentiators

- **Built by a practitioner** — not a generic ML platform. Architecture mirrors actual bank MRM frameworks.
- **Audit-ready by design** — every model output is logged with lineage, version, and timestamp.
- **Explainable by default** — SHAP explanations baked in, not bolted on.
- **Fintech-native** — lightweight deployment (Docker/API) vs. legacy bank infrastructure.

---

## 📬 Contact

**Rajan Puri** 
- 📧 purirajan.rp@gmail.com
- 🌐 rajanpuri.com
- 💼 linkedin.com/in/razanpuri

---

*Licensed under MIT. Built for fintechs who are serious about credit risk.*
