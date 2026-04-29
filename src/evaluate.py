import numpy as np


def rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error (RMSE).
    This measures average prediction error.
    """
    # Convert inputs to numpy arrays
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    # Calculate squared differences
    squared_errors = (y_true - y_pred) ** 2

    # Calculate mean of squared errors
    mean_squared_error = np.mean(squared_errors)

    # Take square root
    rmse_value = np.sqrt(mean_squared_error)

    return float(rmse_value)


def nasa_score(y_true, y_pred):
    """
    NASA scoring function.
    Penalizes late predictions more than early ones.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    # Difference between prediction and true value
    d = y_pred - y_true

    scores = []

    for value in d:
        if value < 0:
            # Early prediction (less penalty)
            score = np.exp(-value / 13.0) - 1.0
        else:
            # Late prediction (more penalty)
            score = np.exp(value / 10.0) - 1.0

        scores.append(score)

    total_score = sum(scores)

    return float(total_score)


def report(y_true, y_pred, label=""):
    """
    Print RMSE and NASA score.
    """
    r = rmse(y_true, y_pred)
    s = nasa_score(y_true, y_pred)

    print(label.rjust(20), " RMSE =", round(r, 2), " NASA score =", round(s, 2))

    return {
        "label": label,
        "rmse": r,
        "nasa_score": s
    }