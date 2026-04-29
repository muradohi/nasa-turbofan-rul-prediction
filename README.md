# Turbofan Engine RUL Prediction

![CI](https://github.com/muradohi/nasa-turbofan-rul-prediction/actions/workflows/ci.yml/badge.svg)

End-to-end predictive maintenance project for estimating Remaining Useful Life (RUL) of turbofan engines using NASA’s CMAPSS dataset.

---

## 🚀 Overview

This project builds a complete machine learning pipeline for **predictive maintenance**, using multivariate sensor time-series data from simulated turbofan engines.

The system models degradation patterns over time and provides actionable insights through an interactive dashboard for fleet-level monitoring.

---

## 🔧 What it does

- Loads multivariate time-series data from 100+ run-to-failure engine trajectories  
- Engineers degradation-aware features (rolling statistics, trends, variance)  
- Trains and evaluates regression models for RUL prediction  
- Evaluates performance using **RMSE** and the **NASA asymmetric scoring function**  
- Visualizes predictions and sensor behavior through a **Plotly Dash fleet-monitoring dashboard**  

---

## Project Structure

```text
nasa_proj/
├── src/                  # Data loaders, features, evaluation, training
├── tests/               # Pytest unit tests
├── notebooks/           # Exploratory analysis and experiments
├── data/
│   └── raw/             # NASA CMAPSS .txt files (not committed to git)
└── .github/
    └── workflows/       # CI/CD pipeline configuration (GitHub Actions)
```

## 📊 Results (FD001)

| Model    | Test RMSE | NASA Score |
|----------|----------:|-----------:|
| Ridge    |   ~20     |    ~1100   |
| XGBoost  |   ~17     |    ~600    |

XGBoost shows improved performance by better capturing nonlinear degradation patterns in sensor data.

---


## Quickstart

```bash
git clone https://github.com/muradohi/nasa-turbofan-rul-prediction.git
cd nasa_proj
uv sync
# put NASA CMAPSS .txt files into data/raw/
uv run python main.py
```
## 🖥️ Dashboard

An interactive dashboard enables:

- Per-engine sensor visualization over time  
- Predicted Remaining Useful Life (RUL) display  
- Risk-based engine status (high / medium / low)  
- Fleet-level monitoring for maintenance prioritisation  

Run locally:

```bash
uv run python dashboard/app.py

## Tech stack

Python 3.11, uv, pandas, scikit-learn, XGBoost, PyTorch (LSTM phase), Plotly Dash, pytest, GitHub Actions.

## Dataset

[NASA CMAPSS Run-to-Failure Simulation](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps).

