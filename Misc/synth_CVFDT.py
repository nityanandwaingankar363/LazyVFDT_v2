# cvfdt.py
import numpy as np
from collections import deque
from sklearn.model_selection import ParameterGrid
from cvfdt import SlidingVFDT


# class Node:
#     id_counter = 0

#     def __init__(
#         self, feature=None, threshold=None, left=None, right=None, *, value=None
#     ):
#         self.id = Node.id_counter
#         Node.id_counter += 1
#         self.feature = feature
#         self.threshold = threshold
#         self.left = left
#         self.right = right
#         self.value = value
#         self.stats = {"class_counts": {}, "total_samples": 0}

#     def is_leaf(self):
#         return self.value is not None

#     def split(self, feature, threshold):
#         self.feature = feature
#         self.threshold = threshold
#         self.left = Node()
#         self.right = Node()

#     def prune(self):
#         self.feature = None
#         self.threshold = None
#         self.left = None
#         self.right = None
#         self.value = max(self.stats["class_counts"], key=self.stats["class_counts"].get)

# class CVFDT:
#     def __init__(self, delta=0.01, window_size=1000):
#         """
#         Initialize the CVFDT algorithm.
#         :param delta: Confidence parameter for Hoeffding bound.
#         :param window_size: Size of the sliding window.
#         """
#         self.delta = delta
#         self.window_size = window_size
#         self.tree = Node()  # Initialize with a root node
#         self.sliding_window = deque(maxlen=window_size)

#     def predict(self, X):
#         """
#         Predict the label for a single instance.
#         :param X: Feature array.
#         :return: Predicted label.
#         """
#         current_node = self.tree
#         while not current_node.is_leaf():
#             feature = current_node.feature
#             threshold = current_node.threshold
#             if feature is None or threshold is None:
#                 break  # In case of incomplete split setup
#             current_node = (
#                 current_node.left if X[feature] <= threshold else current_node.right
#             )
#         # Return the most frequent label if statistics are available
#         if current_node.stats["class_counts"]:
#             return max(
#                 current_node.stats["class_counts"],
#                 key=current_node.stats["class_counts"].get,
#             )
#         else:
#             return None  # No prediction if stats are empty

#     def hoeffding_bound(self, n):
#         """
#         Calculate Hoeffding bound.
#         :param n: Number of observations.
#         :return: Hoeffding bound value.
#         """
#         return np.sqrt(np.log(1 / self.delta) / (2 * n))

#     def update(self, X, y):
#         """
#         Update the decision tree with new instances.
#         :param X: Features.
#         :param y: Labels.
#         """
#         self.sliding_window.append((X, y))
#         # Perform updates on the tree using the sliding window
#         self._update_tree()

#     def _update_tree(self):
#         """
#         Update the decision tree structure based on the current sliding window.
#         This includes recalculating statistics, checking Hoeffding bounds,
#         and adjusting nodes (splitting or pruning) dynamically.
#         """
#         # Track statistics for each node
#         node_statistics = {}

#         for X, y in self.sliding_window:
#             current_node = self.tree
#             while not current_node.is_leaf():
#                 feature = current_node.feature
#                 threshold = current_node.threshold
#                 if feature is None or threshold is None:
#                     break  # Avoid comparing None with numeric values
#                 current_node = (
#                     current_node.left if X[feature] <= threshold else current_node.right
#                 )
#             current_node.stats["total_samples"] += 1
#             current_node.stats["class_counts"][y] = (
#                 current_node.stats["class_counts"].get(y, 0) + 1
#             )

#         # Adjust the tree based on Hoeffding bounds
#         for node_id, stats in node_statistics.items():
#             node = self.tree.find_node_by_id(node_id)
#             hoeffding_bound = self.hoeffding_bound(stats["total_samples"])
#             # Add logic for splitting or pruning based on statistics

# synthetic_data.py
import random


def generate_synthetic_data(n_samples=1000, n_features=5, drift_point=None):
    """
    Generate synthetic data with an optional drift.
    :param n_samples: Number of data points to generate.
    :param n_features: Number of features.
    :param drift_point: Point at which concept drift occurs.
    :return: Generated features and labels.
    """
    X, y = [], []
    for i in range(n_samples):
        # Generate data points before drift
        if drift_point and i >= drift_point:
            # Change in data distribution (example: mean shift)
            features = [random.gauss(5, 1) for _ in range(n_features)]
        else:
            features = [random.gauss(0, 1) for _ in range(n_features)]

        # Generate label (e.g., sum threshold classification)
        label = 1 if sum(features) > n_features * 2.5 else 0
        X.append(features)
        y.append(label)
    return np.array(X), np.array(y)


# example_usage.py
# from cvfdt import CVFDT
# from synthetic_data import generate_synthetic_data

# Initialize the CVFDT model
# cvfdt_model = CVFDT(delta=0.01, window_size=500)

# Generate synthetic data with a concept drift at sample 1000
X, y = generate_synthetic_data(n_samples=6000, n_features=10, drift_point=1000)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# # Simulate streaming the data into the CVFDT model
# print("Streaming data into CVFDT model...")
# for i, (features, label) in enumerate(zip(X_train, Y_train)):
#     cvfdt_model.update(features, label)
#     if i % 10 == 0:
#         print(f"Processed {i} samples.")
#         # predicted_label = cvfdt_model.predict(features)

# print("Data streaming completed.")


# # Predict on a new dataset
# # test_X, test_y = generate_synthetic_data(
# #     n_samples=100, n_features=10
# # )  # Generate test data

# correct = 0
# for features, true_label in zip(X_test, Y_test):
#     predicted_label = cvfdt_model.predict(features)
#     print(f"Features: {features}, Predicted: {predicted_label}, True: {true_label}")
#     if predicted_label == true_label:
#         correct += 1

# accuracy = correct / len(Y_test)
# print(f"Prediction accuracy: {accuracy*100:.2f}")


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# param_grid = {
#     # "delta": [_ for _ in range(0, 1, 0.01)],
#     "delta": [0.1, 0.01, 0.001],
#     "window_size": [_ for _ in range(1, 500, 20)],
# }

param_grid = {
    # "delta": [_ for _ in range(0, 1, 0.01)],
    "delta": [0.1, 0.01, 0.001],
    "window_size": [1, 10, 100, 1000],
}


# Initialize variables to track the best parameters and accuracy
best_accuracy = 0
best_params = None

# Iterate over all parameter combinations in the grid
for params in ParameterGrid(param_grid):
    # Create a new decision tree with the current parameters
    # tree = StreamingDecisionTree(**params)
    cvfdt_model = SlidingVFDT(**params)

    # Update the tree with the training data
    for i in range(len(X_train)):
        cvfdt_model.update(X_train[i], Y_train[i])

    # Evaluate the tree's accuracy on the test set
    correct_predictions = 0
    total_predictions = len(X_test)

    for i in range(total_predictions):
        prediction = cvfdt_model.predict(
            X_test[i]
        )  # Get the prediction for the test sample
        if prediction == Y_test[i]:  # Compare with the actual label
            correct_predictions += 1

    # Calculate accuracy for the current parameter combination
    accuracy = correct_predictions / total_predictions

    # Update the best parameters and accuracy if the current accuracy is higher
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = params


# Print the best parameters and the corresponding accuracy
print(
    f"Synth SlidingVFDT _ Best Parameters: {best_params}\n Best Accuracy: {best_accuracy * 100:.2f}%"
)
