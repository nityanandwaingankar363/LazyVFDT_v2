import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature #`Index of feature to split on`
        self.threshold = threshold #`Threshold value for the feature`
        self.left = left # Left child
        self.right = right # Right child
        self.value = value # For leaf nodes (empty means not a leaf node)
        self.stats = None  # To track statistics for streaming updates

    def is_leaf_node(self):
        return self.value is not None #`Check if a node is a leaf node`


class StreamingDecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, grace_period=200, n_features=None, delta=0.05):
        self.min_samples_split = min_samples_split # Minimum samples required to split a node
        self.max_depth = max_depth  # Maximum depth of the tree
        self.n_features = n_features # Number of features to consider for splits
        self.grace_period = grace_period  # Determines when splits are considered
        self.root = None # Root node of the tree
        self.data_count = 0  # Keeps track of the number of seen data points
        self.decay_rate = 0.75  # Decay rate for time-decay
        self.delta = delta  # Add delta attribute
    def _update_tree(self, node, X, y, depth):
        if node.is_leaf_node():
            if node.stats is None:
                node.stats = Counter()
            node.stats.update({y: 1})

            # Implement time-decay for concept drift
            for key in list(node.stats.keys()):
                node.stats[key] *= self.decay_rate

            if self.data_count % self.grace_period == 0:
                self._attempt_split(node, X, y, depth)
            return

    def _initialize_tree(self, X_sample, y_sample):
        """Initialize the root node."""
        self.n_features = X_sample.shape[0] if not self.n_features else min(X_sample.shape[0], self.n_features)
        self.root = Node(value=self._most_common_label([y_sample]))
        
    def prune_tree(self, node):
        if node.is_leaf_node():
            return
        if node.left and not node.left.is_leaf_node():
            self.prune_tree(node.left)
        if node.right and not node.right.is_leaf_node():
            self.prune_tree(node.right)

        # Prune if both children are leaves and statistics are outdated
        if node.left.is_leaf_node() and node.right.is_leaf_node():
            if not node.left.stats and not node.right.stats:
                node.feature, node.threshold, node.left, node.right = None, None, None, None



    def update(self, X, y):
        """
        Updates the decision tree with a new data point.
        """
        self.data_count += 1

        # Initialize tree if it's the first data point
        if self.root is None:
            self._initialize_tree(X, y)

        # Traverse and update the tree
        self._update_tree(self.root, X, y, depth=0)

    # def _update_tree(self, node, X, y, depth):
    #     """
    #     Traverse the tree and update statistics or split nodes.
    #     """
    #     if node.is_leaf_node():
    #         # Update the statistics for leaf nodes
    #         if node.stats is None:
    #             node.stats = Counter()
    #         node.stats[y] += 1

    #         # Check if node should split
    #         if self.data_count % self.grace_period == 0:
    #             self._attempt_split(node, X, y, depth)
    #         return

        # # Navigate the tree
        # if X[node.feature] <= node.threshold:
        #     if not node.left:
        #         node.left = Node(value=self._most_common_label([y]))
        #     self._update_tree(node.left, X, y, depth + 1)
        # else:
        #     if not node.right:
        #         node.right = Node(value=self._most_common_label([y]))
        #     self._update_tree(node.right, X, y, depth + 1)

    def _attempt_split(self, node, X, y, depth):
        if depth >= self.max_depth or len(node.stats) < self.min_samples_split:
            return

        feat_idxs = np.random.choice(range(len(X)), self.n_features, replace=False)  # Ensure feat_idxs is a list of integers
        best_feature, best_thresh, best_gain = None, None, -1
        second_best_gain = -1  # Initialize second_best_gain
        for feat_idx in feat_idxs:
            thresholds = [X[feat_idx]]
            for thr in thresholds:
                gain = self._information_gain(list(node.stats.elements()), [X[feat_idx]], thr)
                if gain > best_gain:
                    second_best_gain = best_gain  # Update second_best_gain
                    best_gain, best_feature, best_thresh = gain, feat_idx, thr

        if best_feature is not None:
            # Hoeffding bound check
            epsilon = np.sqrt(np.log(1 / self.delta) / (2 * self.data_count))
            if best_gain - second_best_gain > epsilon:
                node.feature = best_feature
                node.threshold = best_thresh
                node.value = None
                node.left = Node(value=self._most_common_label([y]))
                node.right = Node(value=self._most_common_label([y]))
                node.stats = None

    def _best_split(self, X, y, feat_idxs):
        """
        Find the best split point among a subset of features.
        """
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[feat_idx]
            thresholds = [X_column]  # Single data point threshold

            for thr in thresholds:
                # Calculate information gain
                gain = self._information_gain(y, [X_column], thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold

    def _information_gain(self, y, X_column, threshold):
        # Parent entropy
        parent_entropy = self._entropy(y)

        # Create children
        left_idxs = [i for i, val in enumerate(X_column) if val <= threshold]
        right_idxs = [i for i, val in enumerate(X_column) if val > threshold]

        if not left_idxs or not right_idxs:
            return 0

        # Calculate weighted avg. entropy of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l = self._entropy([y[i] for i in left_idxs])
        e_r = self._entropy([y[i] for i in right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # Calculate information gain
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def old_predict(self, X):
        """
        Predicts a single data point by traversing the tree.
        """
        return self._traverse_tree(X, self.root)

    def _traverse_tree(self, X, node):
        """
        Recursively navigate the tree to find the prediction.
        """
        if node.is_leaf_node():
            return node.value

        if X[node.feature] <= node.threshold:
            return self._traverse_tree(X, node.left)
        return self._traverse_tree(X, node.right)
        
    def print_tree(self, node=None, depth=0):
        """
        Recursively prints the decision tree structure.
        
        Args:
            node (Node): The current node to print. If None, starts from the root.
            depth (int): The current depth in the tree, used for indentation.
        """
        if node is None:
            node = self.root  # Start from the root if no node is provided

        indent = "  " * depth  # Indentation based on depth

        # Check if the current node is a leaf node
        if node.is_leaf_node():
            print(f"{indent}Leaf: Predict={node.value}, Stats={dict(node.stats) if node.stats else {}}")
            return

        # Print current node's feature and threshold
        print(f"{indent}Node: Feature={node.feature}, Threshold={node.threshold}")

        # Recurse into left and right subtrees if they exist
        if node.left:
            print(f"{indent}  Left:")
            self.print_tree(node.left, depth + 1)
        if node.right:
            print(f"{indent}  Right:")
            self.print_tree(node.right, depth + 1)

    def predict(self, X, node=None, depth=0, grace_period=200, delta=0.05):
        """
        Predict a label for a single data point X, with Lazy Pruning.
        
        Args:
            X (array-like): The data point to predict.
            node (Node): The current node to evaluate. Defaults to root.
            depth (int): The depth of the current node in the tree.
            grace_period (int): The minimum number of samples required to consider pruning.
            delta (float): Confidence parameter for Hoeffding Bound.

        Returns:
            int: Predicted label for the data point X.
        """
        if node is None:
            node = self.root  # Start prediction from the root

        # If the node is a leaf, return its prediction
        if node.is_leaf_node():
            return node.value

        # Update node stats for Lazy Pruning
        if node.stats is None:
            node.stats = {"sample_count": 0, "misclassification_count": 0}

        # Track samples routed through this branch
        node.stats["sample_count"] += 1

        # Evaluate the prediction accuracy at this node
        predicted_label = node.value
        actual_label = X[-1]  # Assuming last element in X is the label
        if predicted_label != actual_label:
            node.stats["misclassification_count"] += 1

        # Check if pruning is necessary (after grace period)
        if node.stats["sample_count"] >= grace_period:
            misclassification_rate = node.stats["misclassification_count"] / node.stats["sample_count"]

            # Calculate Hoeffding Bound
            epsilon = np.sqrt(np.log(1 / delta) / (2 * node.stats["sample_count"]))

            # If utility is negligible (difference within epsilon), prune
            if misclassification_rate < epsilon:
                # Collapse the branch into a leaf node
                node.value = self._most_common_label(list(node.stats.elements()))
                node.left = None
                node.right = None
                node.feature = None
                node.threshold = None
                node.stats = None
                return node.value

        # Continue traversal based on the feature and threshold
        if X[node.feature] <= node.threshold:
            return self.predict(X, node=node.left, depth=depth + 1, grace_period=grace_period, delta=delta)
        else:
            return self.predict(X, node=node.right, depth=depth + 1, grace_period=grace_period, delta=delta)


import numpy as np

# Simulated streaming data (features and labels)
# streaming_data = [
#     (np.array([2.0, 3.0]), 0),
#     (np.array([1.0, 2.5]), 0),
#     (np.array([3.5, 1.0]), 1),
#     (np.array([3.0, 2.0]), 1),
#     (np.array([1.5, 2.5]), 0),
#     (np.array([2.0, 3.5]), 0),
#     (np.array([0.0, 2.5]), 0),
#     (np.array([3.5, 1.5]), 1),
#     (np.array([3.0, 2.5]), 1),
#     (np.array([1.5, 3.0]), 0),
#     (np.array([1.5, 3.0]), 0),
#     (np.array([0.5, 2.5]), 0),
#     (np.array([3.0, 1.0]), 1),
#     (np.array([2.5, 2.0]), 1),
#     (np.array([1.0, 2.5]), 0),
# ]

# Initialize the StreamingDecisionTree
# tree = StreamingDecisionTree(min_samples_split=2, max_depth=3, grace_period=2, n_features=2)

# Simulate streaming data processing
# print("Updating tree with streaming data...\n")
# for i, (X, y) in enumerate(streaming_data):
#     print(f"Data Point {i+1}: X={X}, y={y}")
#     tree.update(X, y)

# # tree.print_tree()
# # Predict on new data points
# test_data = [
#     np.array([2.5, 3.0]),
#     np.array([3.5, 1.5]),
#     np.array([1.0, 2.0]),
# ]

# print("\nMaking predictions on new data:")
# for i, X in enumerate(test_data):
#     prediction = tree.predict(X)
#     print(f"Test Point {i+1}: X={X} => Prediction: {prediction}")


####### MONKS DATA SET ########
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 


print("Updating tree with streaming data...\n")


###### COLLECTing THE DATA ###############\

monk_s_problems = fetch_ucirepo(id=70) 
  
# data (as pandas dataframes) 
X = monk_s_problems.data.features 
Y = monk_s_problems.data.targets 

import numpy as np

print("Updating tree with streaming data...\n")
# Initialize arrays
new_X = np.empty((0, X.shape[1]))  # Initialize empty 2D array with correct column size
new_Y = np.array([])  # Initialize empty 1D array
tree = StreamingDecisionTree(min_samples_split=34, max_depth=1000, grace_period=6, n_features=6)

# Process X
for x1, x2 in X.iterrows():
    new_X = np.vstack([new_X, x2.to_numpy()])  # Update new_X with rows
    print(f"Data Point type: {type(x2.to_numpy())}")
    print(f"Data Point: {x2.to_numpy()}")
    print(f"new_X:\n{new_X}\n")

# Process Y
for y1, y2 in Y.iterrows():
    new_Y = np.append(new_Y, y2[0])  # Append values to new_Y
    print(f"Data Point type: {type(y2[0])}")
    print(f"Data Point: {y2[0]}")
    print(f"new_Y:\n{new_Y}\n")
###### COLLECTED THE DATA ###############\

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    new_X, new_Y, test_size=0.3, random_state=42, stratify=Y
)

# Print the shapes of the resulting datasets
# print(f"X_train shape: {X_train.shape}")
# print(f"X_test shape: {X_test.shape}")
# print(f"Y_train shape: {Y_train.shape}")
# print(f"Y_test shape: {Y_test.shape}")

## START TRAINING THE TREE ##############\

# print("Updating tree with streaming data...\n")

# for i in range(len(X_train)):
#     tree.update(X_train[i], Y_train[i])


# #STARTE PREDICTION ###############
# # Initialize counters
# correct_predictions = 0
# total_predictions = len(X_test)

# # Iterate over test data
# for i in range(total_predictions):
#     prediction = tree.predict(X_test[i])  # Get the prediction
#     if prediction == Y_test[i]:  # Compare with the actual label
#         correct_predictions += 1

# # Calculate accuracy
# accuracy = correct_predictions / total_predictions
# print(f"Accuracy: {accuracy * 100:.2f}%")

#################################################################
from sklearn.model_selection import ParameterGrid


# val = 

param_grid = {
    'min_samples_split': [1, 2, 5, 8, 10,15, 20, 30, 50],
    'max_depth': [1,3,5,7, 10, 50, 100, 100 , 1000],
    'grace_period': [2, 5, 10, 12, 15, 18, 20],
    'n_features': [1, 2, 3, 4,5, X.shape[1]],
}

best_accuracy = 0
best_params = None

for params in ParameterGrid(param_grid):
    tree = StreamingDecisionTree(**params)
    # Train and evaluate the tree (use your train-test split)
    for i in range(len(X_train)):
        tree.update(X_train[i], Y_train[i])

    correct_predictions = 0
    total_predictions = len(X_test)

    # Iterate over test data
    for i in range(total_predictions):
        prediction = tree.predict(X_test[i])  # Get the prediction
        if prediction == Y_test[i]:  # Compare with the actual label
            correct_predictions += 1

    # Calculate accuracy
    accuracy = correct_predictions / total_predictions
    

    # Calculate accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = params

print(f"Best Parameters: {best_params}\n Best Accuracy: {best_accuracy * 100:.2f}%")


# Best Parameters: {'grace_period': 5, 'max_depth': 5, 'min_samples_split': 1, 'n_features': 3}

# Best Parameters: {'grace_period': 5, 'max_depth': 10, 'min_samples_split': 1, 'n_features': 1}
#  Best Accuracy: 88.46%

#

