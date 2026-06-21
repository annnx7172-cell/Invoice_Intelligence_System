# Invoice Intelligence System

An end-to-end machine learning system that predicts **freight shipping costs** and assesses **invoice risk** for vendor purchase orders, built on a real SQLite procurement database.

**Live App**: [invoiceintellegentsystem.streamlit.app](https://invoiceintellegentsystem-9okgtnwznqrrquao8kgtol.streamlit.app/)

## Overview

This project works with a beverage distributor's procurement database (`inventory.db`), containing purchase orders, vendor invoices, and inventory records. It solves two related but distinct problems:

1. **Freight Cost Prediction (Regression)** — Given an order's details, predict the expected shipping cost.
2. **Invoice Risk Assessment (Classification)** — Given an invoice and its underlying purchase data, flag whether it should be reviewed for potential billing discrepancies or processing delays.

## Data Source

| Table | Rows | Used For | Key Columns |
|---|---|---|---|
| `vendor_invoice` | 5,543 | Freight regression (standalone) | VendorName, Quantity, Dollars, Freight, PODate, InvoiceDate, PayDate |
| `purchases` | 2.37M | Joined into risk classification | PONumber, Brand, Quantity, Dollars, ReceivingDate |

## Approach

### 1. Data Ingestion
Two separate ingestion paths from the same SQLite database:
- **Freight pipeline**: reads `vendor_invoice` directly via `SELECT * FROM vendor_invoice`.
- **Risk pipeline**: joins `vendor_invoice` with an aggregated `purchases` subquery (grouped by `PONumber`) to compare what was invoiced against what was actually purchased and received.

### 2. Exploratory Data Analysis
Key findings that shaped the modeling approach:
- Freight cost correlates **0.985** with order value (Dollars), confirming order value as the dominant driver of shipping cost.
- An initial "global top-25% Freight" risk definition was found to be biased — it disproportionately flagged large vendors (e.g., Diageo, Bacardi) simply because they place larger orders, not because they posed genuine risk.
- This was corrected using **vendor-relative thresholds**, and ultimately replaced with a stronger, join-based definition (see below).

### 3. Feature Engineering
- Date columns converted into day-gap features (`days_po_to_invoice`, `days_invoice_to_pay`, `days_po_to_pay`).
- `Quantity` and `Dollars` log-transformed to correct for heavy right-skew.
- `VendorName` (127+ unique values) encoded using **target encoding** instead of one-hot encoding, computed only on training data to avoid leakage, reducing the freight feature space from 134 columns down to 7.

### 4. Invoice Risk Label
Built from the joined `vendor_invoice` + `purchases` data, an invoice is flagged risky if **either**:
- The invoiced dollar amount differs from the summed item-level purchase dollars by more than $5 (billing mismatch), **or**
- The average receiving delay for that purchase order exceeds 10 days (operational delay)

This produced a balanced target (~33% risky / 67% not risky) based on genuine operational signals, rather than order size alone.

### 5. Model Training
Both pipelines compare multiple algorithms — Linear/Ridge/Lasso, Decision Tree, Random Forest (with OOB scoring), Gradient Boosting, AdaBoost, SVM (SVR/SVC), and a Voting ensemble — tuned via `GridSearchCV`.

**Model selection for risk classification** prioritized **minimizing false negatives** (missed risky invoices) among models with F1 > 0.85, rather than picking purely on highest F1 — since failing to flag a genuinely risky invoice carries greater business cost than over-flagging a safe one.

### 6. Deployment
A two-tab **Streamlit** app serving both models, deployed on Streamlit Community Cloud.

## Results

### Freight Cost Regression (R² Score)

| Model | R² Score |
|---|---|
| Ridge Regression | 0.8716 |
| Lasso Regression | 0.8716 |
| Decision Tree | 0.9631 |
| Random Forest | 0.9712 |
| **Gradient Boosting** | **0.9718** ✅ Selected |
| AdaBoost Regressor | 0.9596 |
| SVR | 0.8702 |
| Voting Regressor | 0.9586 |

Random Forest OOB Score: 0.9652 (closely matches test R², confirming low overfitting)

### Invoice Risk Classification (F1 Score + Confusion Matrix)

| Model | F1 Score | TP | TN | FP | FN |
|---|---|---|---|---|---|
| Logistic Regression | 0.4749 | 137 | 669 | 70 | 233 |
| Decision Tree | 0.9196 | 326 | 726 | 13 | 44 |
| Random Forest | 0.9262 | 320 | 738 | 1 | 50 |
| Gradient Boosting | 0.9021 | 304 | 739 | 0 | 66 |
| AdaBoost Classifier | 0.5490 | 140 | 739 | 0 | 230 |
| SVC | 0.7774 | 248 | 719 | 20 | 122 |
| **Voting Classifier** | **0.9358** ✅ Selected | 328 | 736 | 3 | **42** |

Selected model: **Voting Classifier** (Random Forest + Gradient Boosting + Decision Tree) — chosen for the lowest false-negative count among strong-performing models, balancing missed-risk minimization with F1 performance.

## App Glossary

Terms used in the Streamlit app, explained:

| Term | Meaning |
|---|---|
| **PO (Purchase Order)** | The formal request a buyer sends to a vendor to order goods. Identified by a unique `PONumber`. |
| **PODate** | The date the purchase order was created. |
| **InvoiceDate** | The date the vendor issued the invoice (bill) for the order. |
| **ReceivingDate** | The date the ordered items were physically received. |
| **PayDate** | The date payment was made for the invoice. |
| **Freight** | The shipping/delivery cost associated with an order. |
| **Vendor Name** | The supplier the order was placed with. |
| **Invoice Quantity** | The number of units stated on the vendor's invoice. |
| **Invoice Dollars** | The dollar amount stated on the vendor's invoice. |
| **Total Distinct Brands on PO** | Number of different product brands included in the same purchase order. |
| **Total Item Quantity (from Purchases)** | Sum of quantities across all line items actually recorded in the `purchases` table for that PO — used to cross-check against the invoice. |
| **Total Item Dollars (from Purchases)** | Sum of dollar amounts across all line items actually recorded in `purchases` for that PO — compared against Invoice Dollars to detect billing mismatches. |
| **Avg. Receiving Delay** | Average number of days between the PO date and when items were received, across all line items on that PO. Can be a decimal since it's an average across multiple items. |
| **Days: PO to Invoice** | Number of days between the purchase order date and the invoice date. |
| **Days: Invoice to Pay** | Number of days between the invoice date and the payment date. |
| **Flagged / Risky Invoice** | An invoice identified as warranting manual review due to a billing mismatch or abnormal receiving delay. |

## Tech Stack

- **Language**: Python 3.11
- **ML Libraries**: scikit-learn, pandas, NumPy
- **Database**: SQLite (via Python's built-in `sqlite3`)
- **Web App**: Streamlit
- **Deployment**: Streamlit Community Cloud
- **Version Control**: Git & GitHub

## Project Structure

```
Invoice_Intelligence_System/
├── notebook/
│   ├── Data/
│   │   └── inventory.db          # not tracked in git (size limit)
│   └── 1. EDA.ipynb
├── artifacts/
│   ├── freight_model.pkl
│   ├── freight_preprocessor.pkl
│   ├── freight_vendor_map.pkl
│   ├── risk_model.pkl
│   ├── risk_preprocessor.pkl
│   └── risk_vendor_map.pkl
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
├── app.py
├── requirements.txt
└── README.md
```

## How to Run Locally

```bash
git clone https://github.com/annnx7172-cell/Invoice_Intelligence_System.git
cd Invoice_Intelligence_System

conda create -p venv python=3.11 -y
conda activate ./venv

pip install -r requirements.txt

# Place inventory.db in notebook/Data/ before running ingestion
python -m src.components.data_ingestion

streamlit run app.py
```

## Key Learnings & Design Decisions

- **Label definitions should be questioned, not assumed.** The first risk label (global Freight threshold) looked reasonable but was found to be biased toward large vendors. Validating label quality against the data before training is as important as model selection itself.
- **High-cardinality categoricals need more than One-Hot Encoding.** With 127+ unique vendors, target encoding reduced the feature space dramatically while preserving vendor-specific signal.
- **The "best" model isn't always the highest-scoring one.** For risk classification, minimizing false negatives was prioritized over raw F1, reflecting the real-world cost asymmetry between missing a risky invoice and over-flagging a safe one.

## Author

**Ananya Singh**
GitHub: [@annnx7172-cell](https://github.com/annnx7172-cell)
