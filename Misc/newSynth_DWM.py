import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, ParameterGrid
from dwm import DynamicWeightedMajority


def generate_rotating_hyperplane_with_drift(
    n_samples, n_features, noise=0.1, drift_rate=0.01, random_state=None
):
    rng = np.random.default_rng(random_state)
    weights = rng.uniform(-1, 1, size=n_features)
    bias = 0.0
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples)
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
    return X, y


n_samples = 10000
n_features = 5
noise = 0.1
drift_rate = 0.02
random_state = 47

X, y = generate_rotating_hyperplane_with_drift(
    n_samples, n_features, noise, drift_rate, random_state
)
X = X.reshape(-1, X.shape[-1])
y = y.flatten()
X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

param_grid = {
    "num_classes": [2],
    "beta": [0.2],
    "theta": [0.4],
    "p": [8],
    "create_classifier": [lambda: DecisionTreeClassifier(max_depth=3)],
    "num_features": [X.shape[1]],
    "window_size": [60],
}

# best_accuracy = 0
# best_params = {}
accuracy_over_time = []
processing_times = []

# for params in ParameterGrid(param_grid):
tree = DynamicWeightedMajority(
    num_classes=2,
    beta=0.2,
    theta=0.4,
    p=8,
    create_classifier=lambda: DecisionTreeClassifier(max_depth=3),
    num_features=X.shape[1],
    window_size=60,
)

train_start_time = time.time()
for i in range(len(X_train)):
    tree.update(X_train[i], Y_train[i], i)
train_end_time = time.time()
training_time = train_end_time - train_start_time
correct_predictions = 0
total_predictions = 0

for i in range(len(X_test)):
    start_time = time.time()
    prediction = tree.predict(X_test[i])
    end_time = time.time()
    processing_times.append((end_time - start_time) * 1000)
    if prediction == Y_test[i]:
        correct_predictions += 1
    total_predictions += 1
    accuracy_over_time.append(correct_predictions / total_predictions)
accuracy = correct_predictions / total_predictions
# if accuracy > best_accuracy:
#     best_accuracy = accuracy
#     best_params = params

plt.figure(figsize=(10, 5))
plt.plot(
    range(len(accuracy_over_time)),
    accuracy_over_time,
    label="DWM Accuracy",
    color="blue",
)
plt.xlabel("Number of Streamed Records")
plt.ylabel("Accuracy")
plt.title("Accuracy Over Time (DWM on Rotating Hyperplane)")
plt.legend()
plt.grid(True)
plt.show()

# print(f"\nBest Parameters: {best_params}")
print(f"Final Accuracy: {accuracy * 100:.2f}%")
print(f"Training Time: {training_time:.2f} seconds")
print(f"Average Predicting Time per Record: {np.mean(processing_times):.7f} ms")
