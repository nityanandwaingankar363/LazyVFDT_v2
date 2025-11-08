import time
import os
import csv
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

from dwm import DynamicWeightedMajority
from lazy_decision_tree import LazyDecisionTree
from datasets.electricity import load_electricity
from datasets.gas_sensor import load_gas_sensor
from datasets.airlines import load_airlines


def generate_rotating_hyperplane_with_drift(
    n_samples,
    n_features,
    *,
    noise=0.1,
    drift_rate=0.01,
    random_state=None,
):
    rng = np.random.default_rng(random_state)
    weights = rng.uniform(-1, 1, size=n_features)
    bias = 0.0
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        x = rng.uniform(0, 1, size=n_features)
        boundary = float(np.dot(weights, x))
        label = 1 if boundary >= bias else 0
        if rng.uniform() < noise:
            label = 1 - label
        X[i] = x
        y[i] = label
        weights += rng.uniform(-drift_rate, drift_rate, size=n_features)
    return X, y



def _count_ldt_nodes(tree: LazyDecisionTree) -> int:
    if getattr(tree, "root", None) is None:
        return 0
    count = 0
    stack = [tree.root]
    while stack:
        node = stack.pop()
        if node is None:
            continue
        count += 1
        if node.left is not None:
            stack.append(node.left)
        if node.right is not None:
            stack.append(node.right)
    return count


def evaluate_lazy_tree(
    X_train,
    y_train,
    X_test,
    y_test,
    *,
    seed=47,
    ldt_max_depth=6,
    ldt_grace_period=11,
):
    tree = LazyDecisionTree(
        min_samples_split=1,
        max_depth=ldt_max_depth,
        grace_period=ldt_grace_period,
        n_features=X_train.shape[1],
        seed=seed,
    )
    start = time.perf_counter()
    for x, y in zip(X_train, y_train):
        tree.update(x, y)
    training_time = time.perf_counter() - start

    correct = 0
    latencies = []
    for x, y in zip(X_test, y_test):
        t0 = time.perf_counter()
        pred = tree.predict(x, label=y)
        latencies.append(time.perf_counter() - t0)
        if pred == y:
            correct += 1
    accuracy = correct / len(X_test)
    model_size = _count_ldt_nodes(tree)
    return {
        "model": "Lazy VFDT",
        "accuracy": accuracy,
        "training_time": training_time,
        "latencies": latencies,
        "model_size": model_size,
    }


def evaluate_dwm(
    X_train,
    y_train,
    X_test,
    y_test,
    *,
    seed=47,
    dwm_base_max_depth=3,
    dwm_window_size=60,
    dwm_beta=0.2,
    dwm_theta=0.4,
    dwm_p=8,
):
    # Determine number of classes from labels (supports multi-class datasets like Gas Sensor)
    try:
        n_classes = int(max(np.max(y_train), np.max(y_test))) + 1
    except Exception:
        n_classes = 2
    ensemble = DynamicWeightedMajority(
        num_classes=n_classes,
        beta=dwm_beta,
        theta=dwm_theta,
        p=dwm_p,
        create_classifier=lambda: DecisionTreeClassifier(max_depth=dwm_base_max_depth, random_state=seed),
        num_features=X_train.shape[1],
        window_size=dwm_window_size,
    )
    start = time.perf_counter()
    for i, (x, y) in enumerate(zip(X_train, y_train)):
        ensemble.update(x, y, i)
    training_time = time.perf_counter() - start

    correct = 0
    latencies = []
    for x, y in zip(X_test, y_test):
        t0 = time.perf_counter()
        pred = ensemble.predict(x)
        latencies.append(time.perf_counter() - t0)
        if pred == y:
            correct += 1
    accuracy = correct / len(X_test)
    model_size = len(ensemble.experts)
    return {
        "model": "Dynamic Weighted Majority",
        "accuracy": accuracy,
        "training_time": training_time,
        "latencies": latencies,
        "model_size": model_size,
    }


def _write_csv_row(path, row_dict, *, header_order):
    exists = os.path.exists(path)
    with open(path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header_order)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row_dict.get(k) for k in header_order})


def main():
    parser = argparse.ArgumentParser(description="Compare Lazy VFDT vs DWM on synthetic or real datasets")
    parser.add_argument(
        "--dataset",
        choices=["rotating_hyperplane", "electricity", "gas_sensor", "airlines"],
        default="rotating_hyperplane",
        help="Dataset to run the comparison on",
    )
    parser.add_argument("--data-path", type=str, default=None, help="Path to dataset CSV (required for electricity/gas_sensor)")
    parser.add_argument("--n-samples", type=int, default=10000, help="Synthetic samples (rotating_hyperplane)")
    parser.add_argument("--n-features", type=int, default=5, help="Synthetic features (rotating_hyperplane)")
    parser.add_argument("--noise", type=float, default=0.1, help="Synthetic label noise (rotating_hyperplane)")
    parser.add_argument("--drift-rate", type=float, default=0.02, help="Synthetic drift rate (rotating_hyperplane)")
    parser.add_argument("--seed", type=int, default=47, help="Random seed for synthetic generator (used if --seeds is not provided)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="List of seeds for repeated runs and aggregation")
    parser.add_argument("--csv-out", type=str, default=None, help="Optional path to append per-run metrics as CSV")
    parser.add_argument("--scale", action="store_true", help="Standardize features using StandardScaler (fit on train)")
    parser.add_argument("--scale-per-batch", action="store_true", help="For gas_sensor: standardize per batch using train stats; fallback to global for unseen batches")
    parser.add_argument(
        "--preset",
        choices=["default", "auto", "gas"],
        default="auto",
        help="Parameter preset: default (original), gas (multi-class friendly), or auto by dataset",
    )
    args = parser.parse_args()

    seeds = args.seeds if args.seeds else [args.seed]
    header = [
        "dataset",
        "condition",
        "seed",
        "model",
        "accuracy",
        "training_time_s",
        "avg_latency_ms",
        "p95_latency_ms",
        "model_size",
        "n_features",
        "n_train",
        "n_test",
    ]

    for run_seed in seeds:
        # Prepare data per run (synthetic depends on seed; real data doesn't but we keep consistent runs)
        if args.dataset == "rotating_hyperplane":
            X, y = generate_rotating_hyperplane_with_drift(
                n_samples=args.n_samples,
                n_features=args.n_features,
                noise=args.noise,
                drift_rate=args.drift_rate,
                random_state=run_seed,
            )
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            condition = f"drift_rate={args.drift_rate}"
        elif args.dataset == "electricity":
            if not args.data_path:
                raise SystemExit("--data-path is required for electricity dataset")
            X, y = load_electricity(args.data_path)
            split = int(0.7 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            condition = "chronological_70_30"
        elif args.dataset == "gas_sensor":
            if not args.data_path:
                raise SystemExit("--data-path is required for gas_sensor dataset")
            X, y = load_gas_sensor(args.data_path)
            split = int(0.7 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            condition = "chronological_70_30"
            # Load batch column to enable per-batch scaling; align order with loader (loader sorts by batch)
            batch_series = None
            try:
                df_full = pd.read_csv(args.data_path)
                if "batch" in df_full.columns:
                    df_full = df_full.sort_values(["batch"]).reset_index(drop=True)
                    batch_series = df_full["batch"].to_numpy()
            except Exception:
                batch_series = None
        elif args.dataset == "airlines":
            if not args.data_path:
                raise SystemExit("--data-path is required for airlines dataset")
            X, y = load_airlines(args.data_path)
            split = int(0.7 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            condition = "chronological_70_30"
        else:
            raise SystemExit("Unknown dataset")

        # Optional scaling (fit on train, apply to test)
        if args.scale_per_batch and args.dataset == "gas_sensor" and 'batch_series' in locals() and batch_series is not None:
            # Build per-batch scalers using training data in each batch
            batches_train = batch_series[: len(X_train)]
            batches_test = batch_series[len(X_train) : len(X_train) + len(X_test)]
            unique_batches = np.unique(np.concatenate([batches_train, batches_test]))
            scalers = {}
            # Global fallback scaler if a test batch has no train samples
            global_scaler = StandardScaler().fit(X_train)
            X_train_scaled = np.empty_like(X_train, dtype=float)
            X_test_scaled = np.empty_like(X_test, dtype=float)
            for b in unique_batches:
                idx_train = np.where(batches_train == b)[0]
                idx_test = np.where(batches_test == b)[0]
                if idx_train.size > 0:
                    sc = StandardScaler().fit(X_train[idx_train])
                else:
                    sc = global_scaler
                scalers[int(b)] = sc
                if idx_train.size > 0:
                    X_train_scaled[idx_train] = sc.transform(X_train[idx_train])
                if idx_test.size > 0:
                    X_test_scaled[idx_test] = sc.transform(X_test[idx_test])
            X_train, X_test = X_train_scaled, X_test_scaled
        elif args.scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Choose presets
        ldt_max_depth = 6
        ldt_grace = 11
        dwm_base_depth = 3
        dwm_window = 60
        dwm_beta = 0.2
        dwm_theta = 0.4
        dwm_p = 8

        def apply_gas_preset():
            nonlocal ldt_max_depth, ldt_grace, dwm_base_depth, dwm_window, dwm_beta, dwm_theta, dwm_p
            ldt_max_depth = 10
            ldt_grace = 20
            dwm_base_depth = 4
            dwm_window = 120
            # keep beta/theta/p conservative

        if args.preset == "gas" or (args.preset == "auto" and args.dataset == "gas_sensor"):
            apply_gas_preset()

        lazy_metrics = evaluate_lazy_tree(
            X_train,
            y_train,
            X_test,
            y_test,
            seed=run_seed,
            ldt_max_depth=ldt_max_depth,
            ldt_grace_period=ldt_grace,
        )
        dwm_metrics = evaluate_dwm(
            X_train,
            y_train,
            X_test,
            y_test,
            seed=run_seed,
            dwm_base_max_depth=dwm_base_depth,
            dwm_window_size=dwm_window,
            dwm_beta=dwm_beta,
            dwm_theta=dwm_theta,
            dwm_p=dwm_p,
        )

        results = [lazy_metrics, dwm_metrics]
        for metrics in results:
            latencies = np.array(metrics["latencies"]) if metrics["latencies"] else np.array([np.nan])
            avg_ms = float(latencies.mean() * 1000)
            p95_ms = float(np.nanpercentile(latencies, 95) * 1000)

            print(f"\n{metrics['model']} (seed={run_seed})")
            print("-" * (len(metrics["model"]) + 9))
            print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
            print(f"Training time: {metrics['training_time']:.4f} seconds")
            print(f"Average prediction latency: {avg_ms:.4f} ms")
            print(f"95th percentile latency: {p95_ms:.4f} ms")
            print(f"Model size: {metrics['model_size']}")

            if args.csv_out:
                row = {
                    "dataset": args.dataset,
                    "condition": condition,
                    "seed": run_seed,
                    "model": metrics["model"],
                    "accuracy": metrics["accuracy"],
                    "training_time_s": metrics["training_time"],
                    "avg_latency_ms": avg_ms,
                    "p95_latency_ms": p95_ms,
                    "model_size": metrics["model_size"],
                    "n_features": X_train.shape[1],
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                }
                _write_csv_row(args.csv_out, row, header_order=header)

        improvement = lazy_metrics["accuracy"] - dwm_metrics["accuracy"]
        print(
            "\nRelative accuracy difference (Lazy VFDT - DWM): "
            f"{improvement * 100:.2f} percentage points"
        )


if __name__ == "__main__":
    main()
