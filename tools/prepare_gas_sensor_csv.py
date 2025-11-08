import argparse
import os
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def _natural_key(s: str):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def _read_table(path: Path, sep: Optional[str] = None, encoding: str = "utf-8") -> pd.DataFrame:
    return pd.read_csv(path, sep=sep, engine="python", header=None, encoding=encoding)


def _read_libsvm(path: Path, n_features: Optional[int] = None):
    """
    Read a libsvm-like file: first token is label, followed by index:value pairs.
    Returns (X_dense, y) where X is dense ndarray.
    """
    try:
        from sklearn.datasets import load_svmlight_file
    except Exception as e:
        raise SystemExit("scikit-learn is required for --libsvm parsing. Install scikit-learn.") from e

    X_sparse, y = load_svmlight_file(str(path), n_features=n_features)
    X = X_sparse.toarray()
    # If n_features unspecified and some rows have fewer indices, sklearn infers features; ok
    return X, y


def prepare_from_dir(
    input_dir: Path,
    pattern: str,
    output_path: Path,
    *,
    label_last: bool = True,
    n_features: Optional[int] = None,
    sep: Optional[str] = None,
    encoding: str = "utf-8",
    libsvm: bool = False,
) -> None:
    files = sorted([p for p in input_dir.glob(pattern) if p.is_file()], key=lambda p: _natural_key(p.name))
    if not files:
        raise SystemExit(f"No files match pattern {pattern!r} in {input_dir}")

    parts: List[pd.DataFrame] = []
    inferred_features: Optional[int] = n_features

    for batch_idx, fp in enumerate(files, start=1):
        if libsvm:
            X, y = _read_libsvm(fp, n_features=inferred_features)
            if inferred_features is None:
                inferred_features = X.shape[1]
            cols = [f"x{i}" for i in range(X.shape[1])]
            out = pd.DataFrame(X, columns=cols)
            out["batch"] = batch_idx
            out["label"] = y
            parts.append(out)
        else:
            df = _read_table(fp, sep=sep, encoding=encoding)

            # Infer feature count from first file if not provided
            if inferred_features is None:
                n_cols = df.shape[1]
                if n_cols < 2:
                    raise SystemExit(f"File {fp} has insufficient columns ({n_cols})")
                inferred_features = n_cols - 1  # assume 1 label column

            # Split features/label
            if label_last:
                X = df.iloc[:, : inferred_features]
                y = df.iloc[:, inferred_features]
            else:
                y = df.iloc[:, 0]
                X = df.iloc[:, 1 : 1 + inferred_features]

            # Coerce types and clean label bytes -> str
            X = X.apply(pd.to_numeric, errors="coerce")
            if y.dtype == object:
                try:
                    y = y.apply(lambda v: v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else v)
                except Exception:
                    pass

            # Name columns
            X.columns = [f"x{i}" for i in range(X.shape[1])]
            out = X.copy()
            out["batch"] = batch_idx
            out["label"] = y.values

            parts.append(out)

    merged = pd.concat(parts, axis=0, ignore_index=True)

    # Optional: drop rows with any NaNs in features
    feat_cols = [c for c in merged.columns if c.startswith("x")]
    merged = merged.dropna(subset=feat_cols)

    # Save CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"Wrote {output_path} with shape {merged.shape} (features={len(feat_cols)}, batches={len(files)})")


def prepare_from_file(
    input_file: Path,
    output_path: Path,
    *,
    label_last: bool = True,
    n_features: Optional[int] = None,
    sep: Optional[str] = None,
    encoding: str = "utf-8",
    libsvm: bool = False,
) -> None:
    if libsvm:
        X, y = _read_libsvm(input_file, n_features=n_features)
        cols = [f"x{i}" for i in range(X.shape[1])]
        out = pd.DataFrame(X, columns=cols)
        out["batch"] = 1
        out["label"] = y
        out = out.dropna(subset=[c for c in out.columns if c.startswith("x")])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, index=False)
        print(f"Wrote {output_path} with shape {out.shape} (features={X.shape[1]}, batches=1) from libsvm")
    else:
        df = _read_table(input_file, sep=sep, encoding=encoding)
        n_cols = df.shape[1]
        if n_features is None:
            if n_cols < 2:
                raise SystemExit(f"File {input_file} has insufficient columns ({n_cols})")
            n_features = n_cols - 1

        if label_last:
            X = df.iloc[:, : n_features]
            y = df.iloc[:, n_features]
        else:
            y = df.iloc[:, 0]
            X = df.iloc[:, 1 : 1 + n_features]

        X = X.apply(pd.to_numeric, errors="coerce")
        if y.dtype == object:
            try:
                y = y.apply(lambda v: v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else v)
            except Exception:
                pass

        X.columns = [f"x{i}" for i in range(X.shape[1])]
        out = X.copy()
        out["batch"] = 1
        out["label"] = y.values

        out = out.dropna(subset=[c for c in out.columns if c.startswith("x")])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, index=False)
        print(f"Wrote {output_path} with shape {out.shape} (features={X.shape[1]}, batches=1)")


def main():
    ap = argparse.ArgumentParser(description="Prepare Gas Sensor Array Drift CSV from batch files or a single table")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-dir", type=str, help="Directory containing batch files (e.g., batch1.dat ...)")
    src.add_argument("--input-file", type=str, help="Single input table to convert")
    ap.add_argument("--pattern", type=str, default="batch*.dat", help="Glob pattern within --input-dir (default: batch*.dat)")
    ap.add_argument("--output", type=str, required=True, help="Output CSV path")
    ap.add_argument("--label-last", action="store_true", help="Treat last column as label (default)")
    ap.add_argument("--label-first", dest="label_last", action="store_false", help="Treat first column as label")
    ap.add_argument("--n-features", type=int, default=None, help="Number of feature columns (infer if omitted)")
    ap.add_argument("--sep", type=str, default=None, help="Field separator (infer if omitted)")
    ap.add_argument("--encoding", type=str, default="utf-8", help="File encoding (default: utf-8)")
    ap.add_argument("--libsvm", action="store_true", help="Parse libsvm-like files (label index:value ...)")
    ap.set_defaults(label_last=True)
    args = ap.parse_args()

    output_path = Path(args.output)

    if args.input_dir:
        prepare_from_dir(Path(args.input_dir), args.pattern, output_path,
                         label_last=args.label_last, n_features=args.n_features,
                         sep=args.sep, encoding=args.encoding, libsvm=args.libsvm)
    else:
        prepare_from_file(Path(args.input_file), output_path,
                          label_last=args.label_last, n_features=args.n_features,
                          sep=args.sep, encoding=args.encoding, libsvm=args.libsvm)


if __name__ == "__main__":
    main()
