# Results Summary

## Airlines

| Condition | Model | Runs | Accuracy (%) | Training Time (s) | Avg Latency (ms) | 95th Latency (ms) | Model Size |
| --- | --- | --- | --- | --- | --- | --- | --- |
| chronological_70_30 | Dynamic Weighted Majority | 5 | 52.71 | 1149.97 +/- 757.35 | 0.12 +/- 0.01 | 0.14 +/- 0.01 | 2 |
| chronological_70_30 | Lazy VFDT | 5 | 49.60 +/- 0.01 | 6.19 +/- 0.34 | 0.02 +/- 0.00 | 0.02 +/- 0.00 | 33 |

## Electricity

| Condition | Model | Runs | Accuracy (%) | Training Time (s) | Avg Latency (ms) | 95th Latency (ms) | Model Size |
| --- | --- | --- | --- | --- | --- | --- | --- |
| chronological_70_30 | Dynamic Weighted Majority | 5 | 56.47 +/- 1.20 | 28.09 +/- 0.37 | 0.09 +/- 0.03 | 0.09 +/- 0.03 | 2 |
| chronological_70_30 | Lazy VFDT | 5 | 64.73 +/- 2.21 | 0.31 +/- 0.01 | 0.01 +/- 0.00 | 0.01 +/- 0.00 | 52 |

## Gas Sensor

| Condition | Model | Runs | Accuracy (%) | Training Time (s) | Avg Latency (ms) | 95th Latency (ms) | Model Size |
| --- | --- | --- | --- | --- | --- | --- | --- |
| chronological_70_30 | Dynamic Weighted Majority | 5 | 14.28 +/- 3.74 | 29.68 +/- 1.28 | 0.08 +/- 0.03 | 0.10 +/- 0.03 | 1 |
| chronological_70_30 | Lazy VFDT | 5 | 22.09 +/- 7.11 | 0.97 +/- 0.20 | 0.01 +/- 0.00 | 0.02 +/- 0.00 | 481 |

## Rotating Hyperplane

| Condition | Model | Runs | Accuracy (%) | Training Time (s) | Avg Latency (ms) | 95th Latency (ms) | Model Size |
| --- | --- | --- | --- | --- | --- | --- | --- |
| drift_rate=0.05 | Dynamic Weighted Majority | 5 | 69.29 +/- 10.42 | 7.22 +/- 0.49 | 0.08 +/- 0.03 | 0.08 +/- 0.03 | 1 |
| drift_rate=0.05 | Lazy VFDT | 5 | 75.79 +/- 12.60 | 0.07 +/- 0.00 | 0.01 +/- 0.00 | 0.01 +/- 0.00 | 87 |
