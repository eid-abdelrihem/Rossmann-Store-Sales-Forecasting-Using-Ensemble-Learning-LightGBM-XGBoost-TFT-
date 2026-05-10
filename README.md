<div align="center">

# 🏪 Rossmann Sales Intelligence System

**AI-Powered Sales Forecasting & Conversational Analytics for 1,115 Retail Stores**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.56+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-017CEE?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)


*A production-grade ensemble forecasting system combining Temporal Fusion Transformers (TFT) with XGBoost, orchestrated through a RidgeCV meta-learner, and augmented with LLM-powered conversational intelligence.*

---

[Features](#-features) · [Architecture](#-architecture) · [Results](#-results) 

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [The Forecasting Pipeline](#-the-forecasting-pipeline)
- [Results & Performance](#-results--performance)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)

---

## 🎯 Overview

The **Rossmann Sales Intelligence System** forecasts daily sales for **1,115 European drugstores** using a state-of-the-art ensemble of deep learning and gradient boosting models. It goes beyond static predictions by providing an **AI-powered conversational interface** that enables store managers to ask natural language questions about their store's performance and receive data-grounded, actionable business insights.

### Problem Statement

Rossmann store managers need accurate 6-week sales forecasts to optimize:
- **Inventory management** — Reduce overstock/stockout losses
- **Staff scheduling** — Align labor costs with expected demand
- **Promotion planning** — Maximize ROI on promotional campaigns
- **Holiday preparation** — Plan for demand spikes and closures

### Solution

A **stacking ensemble** of two complementary forecasting paradigms:
1. **Temporal Fusion Transformer (TFT)** — Deep learning with attention-based temporal modeling
2. **XGBoost** — Gradient boosting with engineered tabular features

Combined via a **RidgeCV meta-learner** and served through an interactive **Streamlit dashboard** with **Groq LLM (Llama 3.3 70B)** conversational intelligence.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **Ensemble Forecasting** | TFT + XGBoost stacked ensemble with RidgeCV meta-learner |
| 💬 **Conversational AI** | LLM-powered natural language analytics (Groq/Llama 3.3 70B) |
| 📈 **Interactive Dashboard** | Modern Streamlit UI with Plotly visualizations |
| 🎯 **Per-Store Intelligence** | Individual profiles for all 1,115 stores |
| 📊 **Uncertainty Quantification** | 95% confidence intervals from TFT quantile regression |
| 🏷️ **Promo Impact Analysis** | Quantified promotional lift per store |
| 📅 **Day-of-Week Patterns** | Visual analytics of weekly sales cycles |
| 🔍 **Error Diagnostics** | Color-coded error analysis with threshold alerts |
| 🔐 **Secure API Integration** | Session-based masked API key input |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ROSSMANN SALES INTELLIGENCE                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │   TFT Model  │    │  XGBoost     │    │  RidgeCV Meta    │  │
│  │  (Deep       │───▶│  (Gradient   │───▶│  Learner         │  │
│  │   Learning)  │    │   Boosting)  │    │  (Stacking)      │  │
│  │              │    │              │    │                   │  │
│  │ • Attention  │    │ • 48 Eng.    │    │ • α=0.7152 (TFT) │  │
│  │ • Quantiles  │    │   Features   │    │ • β=0.2894 (XGB) │  │
│  │ • Temporal   │    │ • Non-linear │    │ • 5-Fold CV      │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
│          │                   │                     │            │
│          └───────────────────┴─────────────────────┘            │
│                              │                                  │
│                    ┌─────────▼─────────┐                       │
│                    │  Ensemble Output  │                       │
│                    │  RMSPE = 10.83%   │                       │
│                    └─────────┬─────────┘                       │
│                              │                                  │
│         ┌────────────────────┼────────────────────┐            │
│         ▼                    ▼                    ▼            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐    │
│  │  Streamlit   │  │   Plotly     │  │   Groq LLM      │    │
│  │  Dashboard   │  │   Charts    │  │   (Llama 3.3)    │    │
│  │              │  │             │  │                   │    │
│  │ • KPI Cards  │  │ • Forecast  │  │ • Natural Lang.  │    │
│  │ • Store Nav  │  │ • Error %   │  │ • Data-Grounded  │    │
│  │ • Profiles   │  │ • Day-of-Wk │  │ • Actionable     │    │
│  └──────────────┘  └─────────────┘  └──────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔬 The Forecasting Pipeline

### Phase 1: Exploratory Data Analysis & Assumptions Testing

**Notebook:** `notebooks/01_EDA_Time_Series_Assumptions.ipynb`

| Analysis | Finding |
|----------|---------|
| **Dataset** | 1,017,209 records × 9 features (Jan 2013 – Jul 2015) |
| **Stores** | 1,115 unique stores across 4 types (A, B, C, D) |
| **Weekly Seasonality** | Strong day-of-week effects; most stores closed Sundays |
| **Promotion Impact** | Significant sales uplift during promotion periods |
| **Holiday Effects** | State holidays and school holidays create distinct demand patterns |
| **Baseline Models** | SES → flat-line (failed); Holt's → slight trend only (insufficient) |

**Conclusion:** Classical time series methods cannot capture the multi-variate, non-linear nature of this data. Advanced ML and deep learning approaches are required.

---

### Phase 2: Gradient Boosted Decision Trees

**Notebook:** `notebooks/03_XGBoost_LightGBM.ipynb`

**Feature Engineering Pipeline** (48 features):
- **Temporal:** DayOfWeek, Month, Season, IsWeekend
- **Lag Features:** Sales_Lag_7, Sales_Lag_14, Sales_Lag_28
- **Rolling Statistics:** Sales_Mean_7, Sales_Mean_14, Sales_Std_7
- **Store Aggregations:** store_mean_sales, store_dow_mean_sales
- **Competition:** CompetitionDistance, is_competition_new
- **Promotional:** Promo, promo_yesterday, Promo2 interval features
- **Holiday:** StateHoliday encoding, SchoolHoliday, days_since_holiday, holiday_tomorrow

**Models Trained:**
| Model | Description |
|-------|-------------|
| XGBoost (Default) | Baseline gradient boosting |
| XGBoost (Tuned) | Hyperparameter-optimized via grid search |
| LightGBM (Default) | Baseline histogram-based boosting |
| LightGBM (Tuned) | Optimized for speed and accuracy |

---

### Phase 3: Temporal Fusion Transformer (TFT)

**Notebook:** `notebooks/04_TFT_Model.ipynb`

The TFT is a state-of-the-art deep learning architecture designed specifically for multi-horizon time series forecasting:

| Component | Configuration |
|-----------|---------------|
| **Architecture** | Multi-head attention + GRN + Variable Selection |
| **Input Length** | 30 days lookback |
| **Forecast Horizon** | 6 days ahead |
| **Static Covariates** | StoreType, Assortment |
| **Future Covariates** | DayOfWeek, Month, Promo, Open, Holidays |
| **Quantiles** | [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975] |
| **Training** | 200 epochs, learning_rate=0.001, batch_size=128 |
| **Hardware** | NVIDIA GPU (CUDA 12.6) |

**Key Advantages:**
- **Attention mechanisms** identify which past time steps matter most
- **Variable Selection Networks** learn feature importance automatically
- **Quantile regression** provides calibrated prediction intervals
- **Global model** learns cross-store patterns via static covariates

---

### Phase 4: Stacking Ensemble

**Notebook:** `notebooks/05_Ensemble_Analysis.ipynb`

The final ensemble uses **model stacking** with a RidgeCV meta-learner:

```
Ensemble_Pred = α × TFT_Pred + β × XGB_Pred + intercept
             = 0.7152 × TFT + 0.2894 × XGB − 238.26
```

| Meta-Learner Detail | Value |
|---------------------|-------|
| **Algorithm** | RidgeCV (L2-regularized linear regression) |
| **Cross-Validation** | 5-Fold CV on validation set |
| **TFT Weight (α)** | 0.7152 |
| **XGBoost Weight (β)** | 0.2894 |
| **Intercept** | −238.26 |
| **Regularization** | Automatic alpha selection via built-in CV |

**Why RidgeCV?**
- Handles multicollinearity between TFT and XGBoost predictions
- L2 regularization prevents overfitting to the validation set
- Built-in cross-validation ensures generalization
- Linear combination preserves interpretability

---

## 📊 Results & Performance

### Overall Ensemble Performance

| Metric | TFT (Standalone) | XGBoost (Standalone) | **Ensemble (RidgeCV)** |
|--------|:-----------------:|:--------------------:|:----------------------:|
| **Mean RMSPE** | 11.36% | 19.72% | **10.83%** ✅ |
| **Median RMSPE** | — | — | **10.05%** |
| **Best Store** | — | — | **4.73%** |

### Ensemble Improvement

| Comparison | Stores Where Ensemble Wins | Win Rate |
|------------|:--------------------------:|:--------:|
| Ensemble vs TFT | 763 / 1,115 | **68.4%** |
| Ensemble vs XGBoost | 1,098 / 1,115 | **98.5%** |

### Data Coverage

| Metric | Value |
|--------|-------|
| **Total Stores** | 1,115 |
| **Prediction Period** | Jun 20 – Jul 31, 2015 (42 days) |
| **Total Predictions** | 40,282 daily forecasts |
| **Average Daily Sales** | €6,979 |
| **Average Prediction** | €6,979 (unbiased) |
| **Mean Absolute Error %** | 1.09% (signed, near-zero bias) |

---

## 🛠 Tech Stack

### Core ML & Data

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.11+ | Runtime |
| PyTorch | 2.10 (CUDA 12.6) | TFT deep learning backend |
| Darts | 0.44.1 | Time series framework (TFT implementation) |
| XGBoost | 3.2.0 | Gradient boosted trees |
| LightGBM | 4.6.0 | Histogram-based gradient boosting |
| scikit-learn | 1.8.0 | RidgeCV meta-learner, preprocessing |
| Pandas | 3.0.1 | Data manipulation |
| NumPy | 2.3.5 | Numerical computing |

### Application & Visualization

| Package | Version | Purpose |
|---------|---------|---------|
| Streamlit | 1.56.0 | Interactive web dashboard |
| Plotly | 6.7.0 | Interactive charts |
| Groq | 1.2.0 | LLM API client (Llama 3.3 70B) |

---
