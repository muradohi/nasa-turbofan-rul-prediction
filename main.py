from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from src.data import (
    SENSOR_COLS,
    add_rul_to_train,
    find_constant_sensors,
    load_fd_dataset,
)
from src.evaluate import report
from src.features import compute_rolling_features, get_last_cycle_per_engine


# Paths and settings
DATA_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
DATASET = "FD001"
WINDOW = 30
MAX_RUL = 125
SEED = 42


def main():
    """
    Full pipeline:
    1. Load data
    2. Clean sensors
    3. Feature engineering
    4. Train models
    5. Evaluate
    6. Save predictions
    """

    # -----------------------
    # 1. Load dataset
    # -----------------------
    train_raw, test_raw, rul_test = load_fd_dataset(DATA_DIR, DATASET)

    # -----------------------
    # 2. Remove useless sensors
    # -----------------------
    constant_sensors = find_constant_sensors(train_raw, SENSOR_COLS)
    useful_sensors = []

    for s in SENSOR_COLS:
        if s not in constant_sensors:
            useful_sensors.append(s)

    print("Using", len(useful_sensors), "sensors")
    print("Dropped sensors:", constant_sensors)

    # -----------------------
    # 3. Add RUL to training data
    # -----------------------
    train_labeled = add_rul_to_train(train_raw, max_rul=MAX_RUL)

    # -----------------------
    # 4. Feature engineering
    # -----------------------
    train_features = compute_rolling_features(
        train_raw,
        useful_sensors,
        window=WINDOW
    )

    # Add target (RUL)
    train_features["RUL"] = train_labeled["RUL"].values

    # Select feature columns
    feature_columns = []
    for col in train_features.columns:
        if col not in ["unit", "cycle", "RUL"]:
            feature_columns.append(col)

    # -----------------------
    # 5. Train / validation split by engine
    # -----------------------
    all_units = train_features["unit"].unique()

    train_units, val_units = train_test_split(
        all_units,
        test_size=0.2,
        random_state=SEED
    )

    train_mask = train_features["unit"].isin(train_units)
    val_mask = train_features["unit"].isin(val_units)

    X_train = train_features.loc[train_mask, feature_columns].values
    y_train = train_features.loc[train_mask, "RUL"].values

    X_val = train_features.loc[val_mask, feature_columns].values
    y_val = train_features.loc[val_mask, "RUL"].values

    # -----------------------
    # 6. Scaling (for Ridge only)
    # -----------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # -----------------------
    # 7. Ridge Regression model
    # -----------------------
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train_scaled, y_train)

    ridge_preds = ridge_model.predict(X_val_scaled)
    ridge_preds = np.clip(ridge_preds, 0, MAX_RUL)

    report(y_val, ridge_preds, "Ridge (val)")

    # -----------------------
    # 8. XGBoost model
    # -----------------------
    xgb_model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=SEED,
        n_jobs=-1,
        early_stopping_rounds=30,
    )

    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    xgb_preds = xgb_model.predict(X_val)
    xgb_preds = np.clip(xgb_preds, 0, MAX_RUL)

    report(y_val, xgb_preds, "XGBoost (val)")

    # -----------------------
    # 9. Test set evaluation
    # -----------------------
    test_features_full = compute_rolling_features(
        test_raw,
        useful_sensors,
        window=WINDOW
    )

    test_features = get_last_cycle_per_engine(test_features_full)
    test_features = test_features.merge(rul_test, on="unit")

    X_test = test_features[feature_columns].values
    y_test = test_features["true_RUL"].values

    ridge_test_preds = ridge_model.predict(scaler.transform(X_test))
    ridge_test_preds = np.clip(ridge_test_preds, 0, MAX_RUL)

    xgb_test_preds = xgb_model.predict(X_test)
    xgb_test_preds = np.clip(xgb_test_preds, 0, MAX_RUL)

    print("\n--- Final Test Results ---")
    report(y_test, ridge_test_preds, "Ridge (test)")
    report(y_test, xgb_test_preds, "XGBoost (test)")

    # -----------------------
    # 10. Save results
    # -----------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame({
        "unit": test_features["unit"].values,
        "true_RUL": y_test,
        "ridge_RUL": ridge_test_preds,
        "xgb_RUL": xgb_test_preds,
    })

    output_path = OUT_DIR / f"baseline_predictions_{DATASET}.parquet"
    results.to_parquet(output_path)

    print("Saved results to:", output_path)


if __name__ == "__main__":
    main()