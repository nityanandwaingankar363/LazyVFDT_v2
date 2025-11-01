# Thesis Next Steps Checklist

- [ ] Confirm project scope: lock thesis objectives, research questions, evaluation metrics, and baseline (DWM) expectations.
- [ ] Stabilize codebase: ensure `LazyDecisionTree` and comparison scripts run reproducibly with pinned `requirements.txt`; add minimal smoke tests or sanity checks for streaming updates.
- [ ] Finalize experimental design: decide on datasets/streams (e.g., rotating hyperplane variants, MONK), parameter grids, number of random seeds, and reporting statistics (means, std dev, latency percentiles).
- [ ] Automate experiments: script batch runs that log metrics to CSV/JSON, generate required plots/tables, and archive raw outputs for traceability.
- [ ] Analyze results: interpret Lazy VFDT vs. DWM performance, highlight where the lazy pruning gives advantages, and note limitations or failure cases.
- [ ] Draft thesis chapters: introduction, related work, methodology (algorithm + datasets), experimental setup, results, discussion, conclusion/future work; keep figures/tables cross-referenced.
- [ ] Iterate on writing: integrate advisor feedback, polish graphs and captions, verify citations, and ensure appendices include code/config summaries if required.
- [ ] Prepare submission package: final PDF, source files, ethics or originality forms, and any supplementary materials per university guidelines.
- [ ] Plan defense: craft slide deck, rehearse narrative, anticipate committee questions (algorithm design choices, parameter settings, threats to validity).
- [ ] Execute final checks: run through defense presentation with peers/supervisor, confirm submission deadlines, and back up all materials.

