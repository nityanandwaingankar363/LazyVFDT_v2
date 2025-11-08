import numpy as np
import pandas as pd
from typing import Tuple


def load_electricity(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the Electricity (ELEC2-style) dataset from a CSV file.

    Expected schema (typical MOA export):
        date, day, period, nswprice, nswdemand, vicprice, vicdemand, transfer, class
    where class ∈ {UP, DOWN}.

    If the exact schema differs, the loader will attempt a reasonable fallback by:
        - Using the last column as the label if 'class' is missing
        - One-hot encoding non-numeric features (excluding the label)
        - Preserving original row order if no time columns are present

    Returns:
        X (np.ndarray), y (np.ndarray) ordered chronologically to preserve drift.
    """
    df = pd.read_csv(path)

    # Identify label column
    label_col = None
    for cand in ["class", "Class", "label", "target", "y"]:
        if cand in df.columns:
            label_col = cand
            break
    if label_col is None:
        label_col = df.columns[-1]

    # Preserve original row order (stream order); do not sort by date/day/period

    # Separate label
    y_raw = df[label_col]
    # Drop non-essential time metadata columns that looked off in conversions
    drop_cols = [c for c in ["date", "day"] if c in df.columns]
    X_df = df.drop(columns=[label_col] + drop_cols)

    # Encode categorical features
    X_df = pd.get_dummies(X_df, drop_first=False)

    # Encode label: handle byte-like strings b'UP'/b'DOWN' and map to 1/0
    if y_raw.dtype == object:
        y_str = y_raw.astype(str)
        # Strip Python byte-string artifacts: b'UP' -> UP
        y_clean = y_str.str.replace(r"^b'|^b\"|'$|\"$", "", regex=True).str.upper()
        mapped = y_clean.map({"UP": 1, "DOWN": 0})
        if mapped.isna().any():
            # Fallback to factorize if unexpected labels present
            y, _ = pd.factorize(y_raw)
        else:
            y = mapped.astype(int).to_numpy()
    else:
        # Numeric label
        y = y_raw.to_numpy()

    X = X_df.to_numpy(dtype=float)
    y = np.asarray(y)

    return X, y
