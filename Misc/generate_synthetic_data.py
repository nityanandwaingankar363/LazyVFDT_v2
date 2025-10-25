import numpy as np


# Generate synthetic dataset with concept drift
def generate_synthetic_data(n_samples=1000, n_features=5, drift_point=None):
    X, y = [], []
    for i in range(n_samples):
        if drift_point and i >= drift_point:
            features = [np.random.normal(5, 1) for _ in range(n_features)]
        else:
            features = [np.random.normal(0, 1) for _ in range(n_features)]
        label = 1 if sum(features) > n_features * 2.5 else 0
        X.append(features)
        y.append(label)
    return np.array(X), np.array(y)
