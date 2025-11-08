import numpy as np
import pandas as pd
from typing import Tuple


def load_gas_sensor(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a prepared Gas Sensor Array Drift dataset CSV.

    Expected schema (flexible): features + label column, optionally a 'batch' column
    indicating chronological acquisition (batches capture sensor drift over time).

    Behavior:
        - If 'batch' exists, sort by 'batch' (and index) to preserve drift order.
        - Identify label column among common names, else use the last column.
        - One-hot encode non-numeric features (excluding the label).
        - Return X, y as NumPy arrays ordered chronologically.

    Note: This loader assumes you have pre-converted the original per-batch files
    into a single CSV, or at least a CSV with a 'batch' column. If you have raw
    batch files, merge them in order externally or adapt this loader accordingly.
    """
    df = pd.read_csv(path)

    # Sort by batch if available to preserve drift ordering
    if "batch" in df.columns:
        df = df.sort_values(["batch"]).reset_index(drop=True)

    # Identify label column
    label_col = None
    for cand in ["class", "Class", "label", "target", "y"]:
        if cand in df.columns:
            label_col = cand
            break
    if label_col is None:
        label_col = df.columns[-1]

    y_raw = df[label_col]
    X_df = df.drop(columns=[label_col])

    # Encode categoricals
    X_df = pd.get_dummies(X_df, drop_first=False)

    # Labels to integers
    if y_raw.dtype == object:
        y, _ = pd.factorize(y_raw)
    else:
        y = y_raw.to_numpy()

    X = X_df.to_numpy(dtype=float)
    y = np.asarray(y)

    return X, y

