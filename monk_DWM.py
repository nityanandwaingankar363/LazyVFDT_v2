import numpy as np
import random
import time
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, ParameterGrid
from dwm import DynamicWeightedMajority
from ucimlrepo import fetch_ucirepo

# Fetch dataset with ID 70 (MONKS dataset)
monk_s_problems = fetch_ucirepo(id=70)

# Convert to NumPy arrays
X = monk_s_problems.data.features.to_numpy()
y = monk_s_problems.data.targets.to_numpy().flatten()

# Split into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# # Define hyperparameters
# param_grid = {
#     "num_classes": [2],
#     "beta": [0.4],
#     "theta": [0.2],
#     "p": [4],
#     "create_classifier": [lambda: DecisionTreeClassifier(max_depth=3)],
#     "num_features": [5],
#     "window_size": [50],
# }

accuracy_over_time = []
processing_times = []

# for params in ParameterGrid(param_grid):
tree = DynamicWeightedMajority(
    num_classes=2,
    beta=0.8,
    theta=0.2,
    p=5,
    create_classifier=lambda: DecisionTreeClassifier(max_depth=3),
    num_features=6,
    window_size=50,
)

train_start_time = time.time()

# Update the tree with the training data
for i in range(len(X_train)):
    tree.update(X_train[i], Y_train[i], i)

train_end_time = time.time()
training_time = train_end_time - train_start_time

# Evaluate the model and track accuracy over time
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
    accuracy_over_time.append(correct_predictions / total_predictions)  # Track accuracy

# # Store best accuracy and params
accuracy = correct_predictions / total_predictions


# Plot accuracy over time
plt.figure(figsize=(10, 5))
plt.plot(
    range(len(accuracy_over_time)),
    accuracy_over_time,
    label="DWM Accuracy",
    color="blue",
)
plt.xlabel("Number of Streamed Records")
plt.ylabel("Accuracy")
plt.title("Accuracy Over Time (DWM on Monk Dataset)")
plt.legend()
plt.grid(True)
plt.show()

# Print final results
print(f"Final Accuracy: {accuracy * 100:.2f}%")
print(f"Training Time: {training_time:.2f} seconds")
print(f"Average predicting Time per Record: {np.mean(processing_times):.7f} ms")
