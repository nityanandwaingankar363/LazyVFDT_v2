import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset=["dataset", "condition", "model", "seed"], keep="last")
    return df


def aggregate(df):
    grouped = (
        df.groupby(["dataset", "condition", "model"])
        .agg(
            accuracy=("accuracy", "mean"),
            training_time_s=("training_time_s", "mean"),
            avg_latency_ms=("avg_latency_ms", "mean"),
            p95_latency_ms=("p95_latency_ms", "mean"),
        )
        .reset_index()
    )
    return grouped


def plot_metric(grouped, metric, ylabel, out_path):
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=grouped,
        x="dataset",
        y=metric,
        hue="model",
        palette="muted",
    )
    plt.ylabel(ylabel)
    plt.xlabel("Dataset")
    plt.title(ylabel + " by Dataset")
    plt.xticks(rotation=15)
    plt.legend(title="Model")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Generate plots from results.csv")
    ap.add_argument("--csv", default="results.csv")
    ap.add_argument("--out-dir", default="docs/plots")
    args = ap.parse_args()

    df = load_data(args.csv)
    grouped = aggregate(df)
    out_dir = Path(args.out_dir)
    plot_metric(grouped, "accuracy", "Accuracy", out_dir / "accuracy.png")
    plot_metric(grouped, "training_time_s", "Training Time (s)", out_dir / "training_time.png")
    plot_metric(grouped, "avg_latency_ms", "Average Latency (ms)", out_dir / "avg_latency.png")


if __name__ == "__main__":
    main()
