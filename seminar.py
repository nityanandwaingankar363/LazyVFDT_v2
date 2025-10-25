import numpy as np
import matplotlib.pyplot as plt

def generate_synthetic_data(n_samples=1000, n_features=2, drift_point=None):
    """
    Generate synthetic data with an optional drift using NumPy for consistency.
    :param n_samples: Number of data points to generate.
    :param n_features: Number of features.
    :param drift_point: Point at which concept drift occurs.
    :return: Generated features and labels.
    """
    np.random.seed(42)
    X, y = [], []
    for i in range(n_samples):
        if drift_point is not None and i >= drift_point:
            features = np.random.normal(loc=5, scale=1, size=n_features)
        else:
            features = np.random.normal(loc=0, scale=1, size=n_features)
        label = 1 if np.sum(features) > n_features * 2.5 else 0
        X.append(features)
        y.append(label)
    return np.array(X), np.array(y)

# Generate data
X, y = generate_synthetic_data(n_samples=1000, n_features=2)

# Check class distribution
unique, counts = np.unique(y, return_counts=True)
print("Class distribution:", dict(zip(unique, counts)))

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(X[y==0][:, 0], X[y==0][:, 1], color='lightblue', label='Class 0', alpha=0.6)
plt.scatter(X[y==1][:, 0], X[y==1][:, 1], color='salmon', label='Class 1', alpha=0.6)

# Decision boundary: x1 + x2 = 5 ⇒ x2 = 5 - x1
x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
boundary = 5 - x_vals
plt.plot(x_vals, boundary, color='black', linewidth=2, label='Decision Boundary')

# Formatting
plt.title('Synthetic Data with Decision Boundary')
plt.xlabel('Feature 1 (x1)')
plt.ylabel('Feature 2 (x2)')
plt.legend()
plt.grid(True)
plt.show()
