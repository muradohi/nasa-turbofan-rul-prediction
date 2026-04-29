from pathlib import Path
import pandas as pd

# Define column names
SENSOR_COLS = ["s" + str(i) for i in range(1, 22)]
OP_COLS = ["op1", "op2", "op3"]
ALL_COLS = ["unit", "cycle"] + OP_COLS + SENSOR_COLS


def load_fd_dataset(data_dir, dataset="FD001"):
    """
    Load train, test, and RUL (Remaining Useful Life) data
    from CMAPSS dataset.
    """
    data_dir = Path(data_dir)

    # Load training data
    train_path = data_dir / ("train_" + dataset + ".txt")
    train = pd.read_csv(train_path, sep=r"\s+", header=None)
    train.columns = ALL_COLS

    # Load testing data
    test_path = data_dir / ("test_" + dataset + ".txt")
    test = pd.read_csv(test_path, sep=r"\s+", header=None)
    test.columns = ALL_COLS

    # Load RUL data
    rul_path = data_dir / ("RUL_" + dataset + ".txt")
    rul_test = pd.read_csv(rul_path, sep=r"\s+", header=None)
    rul_test.columns = ["true_RUL"]

    # Add unit column (starts from 1)
    rul_test["unit"] = rul_test.index + 1

    return train, test, rul_test


def add_rul_to_train(train, max_rul=125):
    """
    Add RUL column to training data.
    RUL = max_cycle - current_cycle
    Limit RUL to max_rul value.
    """
    train = train.copy()

    # Find maximum cycle for each unit
    max_cycle_per_unit = train.groupby("unit")["cycle"].max()

    # Create RUL column
    rul_list = []
    for i in range(len(train)):
        unit_id = train.iloc[i]["unit"]
        current_cycle = train.iloc[i]["cycle"]
        max_cycle = max_cycle_per_unit[unit_id]

        rul = max_cycle - current_cycle

        # Apply cap
        if rul > max_rul:
            rul = max_rul

        rul_list.append(rul)

    train["RUL"] = rul_list

    return train


def find_constant_sensors(df, sensor_cols, threshold=1e-6):
    """
    Find sensors with almost no variation (constant values).
    """
    constant_sensors = []

    for col in sensor_cols:
        std_value = df[col].std()

        if std_value < threshold:
            constant_sensors.append(col)

    return constant_sensors