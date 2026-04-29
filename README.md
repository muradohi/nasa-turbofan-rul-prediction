# Turbofan Engine RUL Prediction

![CI](https://github.com/muradohi/turbofan-rul-prediction/actions/workflows/ci.yml/badge.svg)

End-to-end predictive maintenance project: predicting Remaining Useful Life
(RUL) of turbofan jet engines using NASA's CMAPSS dataset.

## What it does

- Loads multivariate sensor time-series for ~100 engines run to failure
- Engineers rolling features (mean, std, slope) per sensor
- Trains Ridge and XGBoost baselines, with an LSTM in progress
- Reports both RMSE and the NASA asymmetric scoring function
- Serves predictions through a Plotly Dash dashboard (in progress)

## Results (FD001)

| Model    | Test RMSE | NASA Score |
|----------|----------:|-----------:|
| Ridge    |   ~20     |    ~1100   |
| XGBoost  |   ~17     |    ~600    |

## Quickstart

```bash
git clone https://github.com/muradohi/nasa-turbofan-rul-prediction.git
cd nasa_proj
uv sync
# put NASA CMAPSS .txt files into data/raw/
uv run python main.py
```

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

## Tech stack

Python 3.11, uv, pandas, scikit-learn, XGBoost, PyTorch (LSTM phase), Plotly Dash, pytest, GitHub Actions.

## Dataset

[NASA CMAPSS Run-to-Failure Simulation](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps).

