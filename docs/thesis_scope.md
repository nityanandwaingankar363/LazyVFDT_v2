# Thesis Scope and Evaluation Plan

## 1) Problem Statement
- Topic: Online/streaming classification under concept drift.
- Contribution: Lazy VFDT (Lazy Decision Tree with pruning at prediction time via Hoeffding-style bound) compared against a standard streaming ensemble baseline (Dynamic Weighted Majority, DWM).
- Goal: Deliver a concise, reproducible evaluation showing Lazy VFDT’s efficiency (training time, latency) with competitive accuracy under common drift patterns.

## 2) Objectives (Locked)
- O1. Stabilize Lazy VFDT with lazy pruning (`lazy_decision_tree/LazyDecisionTree.py`).
- O2. Use DWM with shallow decision trees as the baseline (`dwm/DynamicWeightedMajority.py`).
- O3. Evaluate both on synthetic streams with gradual and sudden drift.
- O4. Report accuracy, training time, and prediction latency (mean and p95) across multiple seeds.
- O5. Produce clear plots/tables and a short narrative supporting the thesis claims.

## 3) Research Questions (Locked)
- RQ1: How does Lazy VFDT’s lazy pruning mechanism influence its tree structure, stability, and memory footprint over time under drift?
- RQ2: How do accuracy, training cost, and prediction latency of Lazy VFDT evolve under gradual vs sudden drift and varying severities?
- RQ3: How sensitive is Lazy VFDT to key hyperparameters (`grace_period`, `max_depth`, `n_features`) and what accuracy–latency trade‑offs do they induce?
- RQ4: Compared to DWM under identical conditions, how does Lazy VFDT fare in accuracy and efficiency (training time and latency)?
- (Optional) RQ5: How do model sizes evolve (Lazy VFDT node count vs DWM expert count) as drift progresses?

## 4) Baselines and Configurations (Locked)
- Lazy VFDT: `min_samples_split=1`, `max_depth=6`, `grace_period=11`, `n_features = d` (all features).
- DWM: experts = `DecisionTreeClassifier(max_depth=3)`, `beta=0.2`, `theta=0.4`, `p=8`, `window_size=60`.
- Rationale: Matches current defaults in `compare_lazy_vs_dwm.py` for immediate reproducibility; avoid broad tuning to keep scope tight.

## 5) Datasets / Streams (Locked)
- Gradual drift: Rotating Hyperplane (implemented) with `n_samples=10000`, `d=5`, `noise=0.1`, `drift_rate ∈ {0.01, 0.02, 0.05}`.
- Sudden drift: Mean-shift generator (see `synth_Lazy.py` / `synth_DWM.py`) with `n_samples=10000`, `d=5`, `noise=0.1`, `drift_point=8000`.
- Split: `train_test_split(test_size=0.3, stratify=y, random_state=42)`; stream test split to collect latency and accuracy-over-time.
- (Optional) Stationary reference: MONK dataset (uses `ucimlrepo`; include only if accessible without friction).

## 6) Evaluation Protocol (Locked)
- For each condition (drift type × severity) and seed:
  1. Generate stream with fixed parameters and `random_state = seed`.
  2. Train on 70% holdout by sequential `update` calls.
  3. Evaluate on 30% test by `predict` (for Lazy VFDT use `predict(x, label=y)`), collecting:
     - Accuracy (overall on test split)
     - Training time (wall-clock seconds)
     - Prediction latency per instance (mean and p95 in ms)
- Seeds: `{13, 29, 47, 83, 101}` (five runs per configuration).
- Aggregation: mean ± std across seeds for each metric.

## 7) Metrics (Locked)
- Primary: Accuracy, Training time (s), Avg prediction latency (ms), p95 latency (ms).
- Secondary (optional): Model size (Lazy VFDT node count; DWM expert count) if trivial to record.

## 8) Success Criteria (Locked)
- Accuracy within ±2 percentage points of DWM while significantly lower training time and average latency for Lazy VFDT.
- Reproducible scripts generate CSVs/plots with aggregated results.

## 9) Out of Scope (Locked)
- Extensive hyperparameter tuning, additional baselines, complex real-world datasets, and advanced prequential protocols beyond the simple holdout + stream simulation.

## 10) Deliverables
- Scripts: extend `compare_lazy_vs_dwm.py` to support seeds, drift severities, and CSV logging.
- Results: CSV files per condition, aggregate summaries, and plots.
- Thesis content: method, setup, results, and discussion referencing produced tables/figures.

## 11) Acceptance Checklist
- [ ] Runs reproduce across 5 seeds for all chosen conditions.
- [ ] CSV logs include accuracy, training_time, avg_latency_ms, p95_latency_ms for both models.
- [ ] Aggregated tables and 2–3 plots (accuracy-over-time; training time/latency bar charts) generated.
- [ ] Narrative explains when/why Lazy VFDT helps and any limitations.
