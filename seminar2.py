import numpy as np
import matplotlib.pyplot as plt

def generate_rotating_hyperplane_with_drift(
    n_samples, n_features, noise=0.1, drift_rate=0.01, random_state=None
):
    rng = np.random.default_rng(random_state)
    weights = rng.uniform(-1, 1, size=n_features)
    bias = 0.0
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples)
    weights_list = []

    for i in range(n_samples):
        x = rng.uniform(0, 1, size=n_features)
        decision_boundary = np.dot(weights, x)
        label = 1 if decision_boundary >= bias else 0
        if rng.uniform(0, 1) < noise:
            label = 1 - label
        X[i] = x
        y[i] = label
        drift = rng.uniform(-drift_rate, drift_rate, size=n_features)
        weights += drift

        # Save weight snapshot at the halfway point for plotting
        if i == n_samples // 2:
            mid_weights = weights.copy()
        if i == n_samples - 1:
            final_weights = weights.copy()

    return X, y, mid_weights, final_weights

# Generate data
n_features = 2
X, y, mid_weights, final_weights = generate_rotating_hyperplane_with_drift(
    n_samples=1000,
    n_features=n_features,
    noise=0.1,
    drift_rate=0.01,
    random_state=42
)

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(X[y==0][:, 0], X[y==0][:, 1], color='lightblue', label='Class 0', alpha=0.6)
plt.scatter(X[y==1][:, 0], X[y==1][:, 1], color='salmon', label='Class 1', alpha=0.6)

# Plot the decision boundary (at halfway and final time)
def plot_hyperplane(weights, label, color):
    w1, w2 = weights
    if w2 != 0:
        x_vals = np.linspace(0, 1, 100)
        y_vals = - (w1 * x_vals) / w2  # because w1*x + w2*y + b = 0 → y = -w1/w2 * x
        plt.plot(x_vals, y_vals, label=label, color=color, linewidth=2)
    else:
        x_val = -0 / w1
        plt.axvline(x=x_val, label=label, color=color, linewidth=2)

plot_hyperplane(mid_weights, 'Midpoint Hyperplane', 'green')
plot_hyperplane(final_weights, 'Final Hyperplane', 'black')

# Formatting
plt.title('Rotating Hyperplane with Concept Drift')
plt.xlabel('Feature 1 (x1)')
plt.ylabel('Feature 2 (x2)')
plt.legend()
plt.grid(True)
plt.xlim(0, 1.5)
plt.ylim(0, 1.5)
plt.show()
