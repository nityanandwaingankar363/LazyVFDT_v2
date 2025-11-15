# Report Preparation Guide

## Dataset Coverage
- **Synthetic Rotating Hyperplane** – binary concept drift with controllable noise/drift rate; demonstrates Lazy VFDT’s out-of-the-box edge on gradual drift.
- **Electricity (ELEC2)** – widely used real drift benchmark; Lazy VFDT shows +8.3 percentage-point accuracy over DWM while training ~90x faster.
- **Gas Sensor Array Drift** – 6-class, heavy drift; requires per-batch scaling and deeper Lazy VFDT to stay competitive, illustrating how preprocessing matters.
- **Airlines** – long-running real stream with noisy labels; tuned DWM edges accuracy but at massive computational cost, letting us emphasize efficiency trade-offs.

These four scenarios cover gradual vs sudden drift, binary vs multi-class, and synthetic vs real-world sources—sufficient scope for the thesis narrative.

## Data To Collect Before Writing
- Final `results.csv` (already generated) plus the aggregated `docs/results_summary.md`.
- Plots derived from `results.csv`:
  - Accuracy per dataset (`Lazy VFDT` vs `DWM`).
  - Training time and latency comparisons.
  - Optional scatter of model size vs accuracy.
- Configuration notes for each dataset:
  - Synthetic/Electricity: default `max_depth=6`, `grace_period=11`, no scaling.
  - Gas Sensor: `--scale-per-batch`, `max_depth=10`, `grace_period=20`.
  - Airlines: `--scale`, `max_depth=10`, `grace_period=20`, DWM `max_depth=4`, `window_size=120`.
- Qualitative observations (kept in lab notebook or added below) explaining why accuracy differs per dataset.

## Suggested Thesis Structure
1. **Introduction** – motivation for streaming classification under drift; introduce Lazy VFDT as a lightweight alternative.
2. **Related Work** – VFDT variants, ensemble baselines (DWM), drift handling methods, sensor drift literature.
3. **Methodology** – architecture of Lazy VFDT (nodes, lazy pruning with Hoeffding bound), baseline description, datasets, preprocessing (scaling/per-batch), hyperparameter policy.
4. **Experimental Setup** – metrics (accuracy, training time, avg/p95 latency, model size), hardware, seeds, scripts/flags.
5. **Results** – per-dataset tables/plots; highlight both accuracy and efficiency; discuss impact of tuning (e.g., scaling or deeper trees for multi-class).
6. **Discussion** – synthesize findings: where Lazy VFDT shines, limitations (multi-class accuracy, need for scaling), computational savings vs ensembles.
7. **Conclusion & Future Work** – recap contributions, outline extensions (adaptive tuning, additional baselines, online hyperparameter selection).

## Narrative Focus
- **Efficiency First**: Across Electricity and synthetic streams, Lazy VFDT matches or surpasses DWM accuracy while training/predicting orders of magnitude faster.
- **Adaptability via Preprocessing**: Gas Sensor shows that per-batch scaling plus deeper trees let Lazy VFDT remain viable on harsh drift, reinforcing the “lazy pruning” story.
- **Trade-offs on Airlines**: Even when DWM edges accuracy, Lazy VFDT still offers huge computational savings—valuable when latency or resource budgets dominate.
- **Practical Guidance**: Document the tuning steps so readers can reproduce improvements; emphasize that small, interpretable adjustments suffice (changing depth, grace period, scaling).

Capturing these elements now (results tables, plots, configuration notes) will make drafting the report straightforward and defensible.

