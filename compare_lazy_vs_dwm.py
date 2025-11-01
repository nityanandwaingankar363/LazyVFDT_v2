import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from dwm import DynamicWeightedMajority
from lazy_decision_tree import LazyDecisionTree


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



def evaluate_lazy_tree(X_train, y_train, X_test, y_test):
    tree = LazyDecisionTree(
        min_samples_split=1,
        max_depth=6,
        grace_period=11,
        n_features=X_train.shape[1],
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
    return {
        "model": "Lazy VFDT",
        "accuracy": accuracy,
        "training_time": training_time,
        "latencies": latencies,
    }


def evaluate_dwm(X_train, y_train, X_test, y_test):
    ensemble = DynamicWeightedMajority(
        num_classes=2,
        beta=0.2,
        theta=0.4,
        p=8,
        create_classifier=lambda: DecisionTreeClassifier(max_depth=3),
        num_features=X_train.shape[1],
        window_size=60,
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
    return {
        "model": "Dynamic Weighted Majority",
        "accuracy": accuracy,
        "training_time": training_time,
        "latencies": latencies,
    }


def main():
    X, y = generate_rotating_hyperplane_with_drift(
        n_samples=10000,
        n_features=5,
        noise=0.1,
        drift_rate=0.02,
        random_state=47,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    lazy_metrics = evaluate_lazy_tree(X_train, y_train, X_test, y_test)
    dwm_metrics = evaluate_dwm(X_train, y_train, X_test, y_test)

    results = [lazy_metrics, dwm_metrics]
    for metrics in results:
        latencies = np.array(metrics["latencies"])
        print(f"\n{metrics['model']}")
        print("-" * len(metrics["model"]))
        print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
        print(f"Training time: {metrics['training_time']:.4f} seconds")
        print(
            "Average prediction latency: "
            f"{latencies.mean() * 1000:.4f} ms"
        )
        print(
            "95th percentile latency: "
            f"{np.percentile(latencies, 95) * 1000:.4f} ms"
        )

    improvement = lazy_metrics["accuracy"] - dwm_metrics["accuracy"]
    print(
        "\nRelative accuracy difference (Lazy VFDT - DWM): "
        f"{improvement * 100:.2f} percentage points"
    )


if __name__ == "__main__":
    main()
