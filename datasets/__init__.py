"""Dataset loader package for real-world drift benchmarks.

Provides lightweight utilities to load and preprocess:
- Electricity (ELEC2-like) dataset
- Gas Sensor Array Drift (UCI variant)

Each loader returns (X, y) as NumPy arrays ordered chronologically to preserve drift.
"""

