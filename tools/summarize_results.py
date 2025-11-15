import argparse
from pathlib import Path
import pandas as pd


def summarize(df):
    grouped = (
        df.groupby(["dataset", "condition", "model"])
        .agg(
            runs=("accuracy", "count"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            training_mean=("training_time_s", "mean"),
            training_std=("training_time_s", "std"),
            latency_mean=("avg_latency_ms", "mean"),
            latency_std=("avg_latency_ms", "std"),
            p95_mean=("p95_latency_ms", "mean"),
            p95_std=("p95_latency_ms", "std"),
            size_mean=("model_size", "mean"),
        )
        .reset_index()
    )
    return grouped


def format_pm(mean, std, scale=1.0):
    if pd.isna(std):
        std = 0.0
    return f"{mean*scale:.2f} +/- {std*scale:.2f}" if std > 0 else f"{mean*scale:.2f}"


def build_markdown(grouped):
    lines = ["# Results Summary", ""]
    for dataset in grouped["dataset"].unique():
        subset = grouped[grouped["dataset"] == dataset]
        lines.append(f"## {dataset.replace('_', ' ').title()}")
        lines.append("")
        lines.append("| Condition | Model | Runs | Accuracy (%) | Training Time (s) | Avg Latency (ms) | 95th Latency (ms) | Model Size |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for _, row in subset.sort_values(["condition", "model"]).iterrows():
            acc = format_pm(row["accuracy_mean"], row["accuracy_std"], scale=100)
            train = format_pm(row["training_mean"], row["training_std"], scale=1)
            lat = format_pm(row["latency_mean"], row["latency_std"], scale=1)
            p95 = format_pm(row["p95_mean"], row["p95_std"], scale=1)
            size = f"{row['size_mean']:.0f}"
            lines.append(
                f"| {row['condition']} | {row['model']} | {int(row['runs'])} | {acc} | {train} | {lat} | {p95} | {size} |"
            )
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Summarize results CSV into markdown")
    parser.add_argument("--csv", default="results.csv")
    parser.add_argument("--out", default="docs/results_summary.md")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=["dataset", "condition", "model", "seed"], keep="last")
    grouped = summarize(df)
    md = build_markdown(grouped)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"Wrote summary to {out_path}")


if __name__ == "__main__":
    main()
