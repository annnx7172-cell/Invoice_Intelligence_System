# 📦 Invoice Intelligence System

> *"Every invoice that slips through unchecked is a potential loss. Every one that gets flagged unnecessarily wastes someone's time. Can a machine find the balance?"*

**Live App** → [invoiceintellegentsystem.streamlit.app](https://invoiceintellegentsystem-9okgtnwznqrrquao8kgtol.streamlit.app/)
**GitHub** → [annnx7172-cell/Invoice_Intelligence_System](https://github.com/annnx7172-cell/Invoice_Intelligence_System)

---

## 🧩 The Problem

Imagine you're running a beverage distribution company. Every month, hundreds of invoices arrive from your vendors — each one claiming a certain quantity was shipped, at a certain price, with a certain freight cost. Most of them are fine. But some aren't.

Maybe a vendor billed you for 500 cases but the warehouse only received 480. Maybe an invoice charges $6,200 but the actual purchase order totals $5,900. Maybe a shipment took two weeks to arrive when it usually takes seven days. Any of these could mean a billing error, a supply chain problem, or worse — overbilling.

Manually checking every invoice against every purchase record is tedious, slow, and expensive. This project builds an automated system that does it for you — flagging the invoices that genuinely need a second look, while letting the clean ones through without friction.

**This is not a toy problem. It's the kind of system logistics companies, procurement teams, and finance departments actually need.**

---

## 💡 What This System Does

This project solves **two problems at once**, using two separate machine learning models:

| Task | Type | Question Answered |
|---|---|---|
| **Freight Cost Prediction** | Regression | "Given this order, how much should the shipping cost?" |
| **Invoice Risk Assessment** | Classification | "Should this invoice be flagged for manual review?" |

Both are served through a single, clean Streamlit interface — no technical knowledge required to use it.

---

## 🏢 Business Context

The dataset comes from a real procurement database (`inventory.db`) belonging to a **beverage distributor**. It contains five interconnected tables covering everything from purchase orders to inventory levels. This project works with two of them:

### Data Sources

| Table | Size | What It Contains |
|---|---|---|
| `vendor_invoice` | 5,543 records | One row per invoice — vendor name, order value, freight cost, payment dates |
| `purchases` | 2.37 million records | Individual line items — every brand, quantity, and price on every purchase order |

The key insight that makes this project interesting: **these two tables tell different stories about the same orders**. The invoice says what the vendor claims was delivered and billed. The purchases table says what was actually ordered and received. When those stories don't match — that's a risk signal.

---

## 🔄 Project Workflow

```
SQLite Database
      ↓
Data Ingestion (two separate pipelines)
      ↓
Exploratory Data Analysis
      ↓
Feature Engineering (date gaps, log transforms, target encoding)
      ↓
Risk Label Construction (billing mismatch + receiving delay)
      ↓
Model Training (8 algorithms × GridSearchCV × two tasks)
      ↓
Model Selection (confusion matrix-driven, not just F1)
      ↓
Streamlit App → Live Deployment
```

---

## 🛠️ Modular Programming

The pipeline is built using a **modular, production-style structure** — each stage (ingestion, transformation, training) lives in its own independent file. This means any component can be updated, replaced, or tested without touching the rest of the system.

---

## 🔍 Key Findings from EDA

- **Freight-Dollars correlation (0.985)** — order value is the single dominant driver of freight cost.
- **Right-skewed distributions** — Quantity and Dollars required log-transformation before modeling.
- **Vendor-biased risk label** — an initial global threshold disproportionately flagged large vendors; corrected using a join-based billing mismatch + receiving delay definition.
- **High-cardinality VendorName** — 127+ unique vendors replaced One-Hot Encoding with Target Encoding, reducing feature space from 134 to 7 columns.

---

## 📊 Results

### Freight Cost Regression — R² Scores

| Model | R² Score |
|---|---|
| SVR | 0.8702 |
| Ridge / Lasso | 0.8716 |
| AdaBoost | 0.9596 |
| Decision Tree | 0.9631 |
| Voting Regressor | 0.9586 |
| Random Forest | 0.9712 |
| **Gradient Boosting** | **0.9718 ✅** |

*Random Forest OOB Score: 0.9652 — closely matching its test R², confirming the model is not overfitting.*

### Invoice Risk Classification — F1 + Confusion Matrix

| Model | F1 | TP | TN | FP | FN |
|---|---|---|---|---|---|
| Logistic Regression | 0.4749 | 137 | 669 | 70 | 233 |
| AdaBoost | 0.5490 | 140 | 739 | 0 | 230 |
| SVC | 0.7774 | 248 | 719 | 20 | 122 |
| Gradient Boosting | 0.9021 | 304 | 739 | 0 | 66 |
| Decision Tree | 0.9196 | 326 | 726 | 13 | 44 |
| Random Forest | 0.9262 | 320 | 738 | 1 | 50 |
| **Voting Classifier** | **0.9358 ✅** | **328** | **736** | **3** | **42** |

*Random Forest OOB Score: 0.9614*

**Why the Voting Classifier was selected over Random Forest (which had a higher F1)?**
The model selection logic prioritized **minimizing False Negatives** (missed risky invoices) among models with F1 > 0.85. In business terms: failing to flag a genuinely risky invoice is far more costly than occasionally flagging a safe one for review. The Voting Classifier caught 328 risky invoices while only missing 42 — fewer missed cases than any other strong model, with just 3 false alarms.

---

## 🧰 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| Database | SQLite (via Python's built-in `sqlite3`) |
| Data Processing | pandas, NumPy |
| Machine Learning | scikit-learn (RF, GBM, SVM, Logistic Regression, Ensembles) |
| Serialization | dill, pickle |
| Web App | Streamlit |
| Deployment | Streamlit Community Cloud |
| Version Control | Git & GitHub |

---

## 🔮 Future Improvements

- **Live database integration** — connect the app directly to a procurement database so it auto-fetches purchase-level details by PO number, making it usable by real operations teams without manual data entry.
- **Operations Research for route and cost optimization** — extend beyond prediction into prescriptive analytics: use linear programming or vehicle routing algorithms to suggest the most cost-efficient shipping routes and vendor order schedules, complementing the freight cost predictions with actionable optimization.
- **Anomaly scoring with vendor dashboards** — replace binary risk flags with a continuous risk probability score, and add vendor-level trend views showing which suppliers consistently generate billing discrepancies or receiving delays over time.

---

## 👩‍💻 Author

**Ananya Singh**
MSc Statistics and Computing (Machine Learning)
GitHub: [@annnx7172-cell](https://github.com/annnx7172-cell)