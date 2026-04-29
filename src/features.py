
import numpy as np
import pandas as pd


def rolling_slope(values):
    """
    Calculate slope of a simple linear trend.
    Positive slope = increasing trend
    Negative slope = decreasing trend
    """
    n = len(values)

    # If not enough data points, return 0
    if n < 2:
        return 0.0

    x = np.arange(n)

    # Calculate averages
    x_mean = x.mean()
    y_mean = np.mean(values)

    # Compute covariance between x and y
    covariance = np.sum((x - x_mean) * (values - y_mean))

    # Compute variance of x
    variance = np.sum((x - x_mean) ** 2)

    # Avoid division by zero
    if variance == 0:
        return 0.0

    return covariance / variance


def compute_rolling_features(df, sensor_cols, window=30):
    """
    Create rolling features for each sensor:
    - mean (average behavior)
    - std (variability / instability)
    - slope (trend direction)
    - last value (current reading)
    """

    # Sort data properly by engine and time
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)

    # Start output dataframe with identifiers only
    features = df[["unit", "cycle"]].copy()

    # Process each sensor separately
    for sensor in sensor_cols:

        # Group data by engine
        grouped = df.groupby("unit")[sensor]

        features[sensor + "_mean"] = grouped.apply(
            lambda x: x.rolling(window, min_periods=1).mean()
        ).reset_index(level=0, drop=True)

        features[sensor + "_std"] = grouped.apply(
            lambda x: x.rolling(window, min_periods=2).std()
        ).reset_index(level=0, drop=True).fillna(0)

        features[sensor + "_slope"] = grouped.apply(
            lambda x: x.rolling(window, min_periods=2).apply(rolling_slope, raw=True)
        ).reset_index(level=0, drop=True).fillna(0)

        # Most recent value (current sensor reading)
        features[sensor + "_last"] = df[sensor].values

    return features


def get_last_cycle_per_engine(features):
    """
    Keep only the last available cycle for each engine.
    Used for test-time prediction.
    """

    # Sort so last cycle is at bottom
    features = features.sort_values(["unit", "cycle"])

    # Keep only final row per engine
    last_cycles = features.groupby("unit").tail(1)

    # Reset index for clean output
    return last_cycles.reset_index(drop=True)