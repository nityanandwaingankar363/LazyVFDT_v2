import numpy as np
import pandas as pd
from typing import Tuple


def load_airlines(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the MOA Airlines / Flight Delay dataset from CSV.

    The CSV typically contains both numeric and categorical columns plus a
    binary target (e.g., 'target', 'delay', 'class'). This loader:
      - Finds a label column among common names, defaulting to the last column.
      - One-hot encodes categorical feature columns (including day-of-week, etc.).
      - Maps string labels (e.g., 'delayed', 'ontime', 'YES', 'NO') to 0/1 if possible,
        otherwise factorizes labels consistently.
      - Returns NumPy arrays ordered as provided in the file.
    """

    df = pd.read_csv(path)

    label_col = None
    for cand in ["delay", "Delay", "class", "Class", "target", "TARGET", "label", "Label", "y"]:
        if cand in df.columns:
            label_col = cand
            break
    if label_col is None:
        label_col = df.columns[-1]

    y_raw = df[label_col]
    X_df = df.drop(columns=[label_col])

    X_df = pd.get_dummies(X_df, drop_first=False)

    if y_raw.dtype == object:
        series = y_raw.astype(str).str.strip().str.lower()
        mapping = {
            "true": 1,
            "false": 0,
            "yes": 1,
            "no": 0,
            "y": 1,
            "n": 0,
            "1": 1,
            "0": 0,
            "delayed": 1,
            "ontime": 0,
            "on-time": 0,
        }
        mapped = series.map(mapping)
        if mapped.isna().any():
            y, _ = pd.factorize(series)
        else:
            y = mapped.to_numpy(dtype=int)
    else:
        y = y_raw.to_numpy()

    X = X_df.to_numpy(dtype=float)
    y = np.asarray(y)

    return X, y

