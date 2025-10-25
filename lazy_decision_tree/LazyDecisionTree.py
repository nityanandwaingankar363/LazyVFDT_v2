import numpy as np
from collections import Counter


class Node:
    def __init__(
        self, feature=None, threshold=None, left=None, right=None, *, value=None
    ):
        self.feature = feature  # `Index of feature to split on`
        self.threshold = threshold  # `Threshold value for the feature`
        self.left = left  # Left child
        self.right = right  # Right child
        self.value = value  # Prediction for leaf nodes
        # Keep track of the majority label routed through this node
        self.majority_label = value
        # Maintain label counts for splitting and majority tracking
        self.split_stats = Counter()
        # Track lazy-pruning statistics separately from split statistics
        self.pruning_stats = {"sample_count": 0, "misclassification_count": 0}

    def is_leaf_node(self):
        return self.left is None and self.right is None


class LazyDecisionTree:
    def __init__(
        self,
        min_samples_split=2,
        max_depth=100,
        grace_period=200,
        n_features=None,
        seed=6,
    ):
        self.min_samples_split = (
            min_samples_split  # Minimum samples required to split a node
        )
        self.max_depth = max_depth  # Maximum depth of the tree
        self.n_features = n_features  # Number of features to consider for splits
        self.grace_period = grace_period  # Determines when splits are considered
        self.root = None  # Root node of the tree
        self.data_count = 0  # Keeps track of the number of seen data points
        self.seed = seed  # Seed for reproducibility

    def _initialize_tree(self, X_sample, y_sample):
        """Initialize the root node."""
        self.n_features = (
            X_sample.shape[0]
            if not self.n_features
            else min(X_sample.shape[0], self.n_features)
        )
        self.root = Node(value=y_sample)
        self.root.majority_label = y_sample

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

    def _update_tree(self, node, X, y, depth):
        """
        Traverse the tree and update statistics or split nodes.
        """
        node.split_stats[y] += 1
        node.majority_label = self._most_common_label(node.split_stats)

        if node.is_leaf_node():
            # Ensure the leaf prediction matches the observed majority label
            node.value = node.majority_label

            # Check if node should split
            if self.data_count % self.grace_period == 0:
                self._attempt_split(node, X, y, depth)
            return

        # Navigate the tree
        if X[node.feature] <= node.threshold:
            if not node.left:
                node.left = Node(value=node.majority_label)
            self._update_tree(node.left, X, y, depth + 1)
        else:
            if not node.right:
                node.right = Node(value=node.majority_label)
            self._update_tree(node.right, X, y, depth + 1)

    def _attempt_split(self, node, X, y, depth):
        """
        Check if a node should split based on its statistics.
        """
        if depth >= self.max_depth or sum(node.split_stats.values()) < self.min_samples_split:
            return
        np.random.seed(self.seed)
        # Randomly sample features for splitting
        feat_idxs = np.random.choice(len(X), self.n_features, replace=False)

        # Find the best split
        best_feature, best_thresh = self._best_split(X, node.split_stats, feat_idxs)

        if best_feature is not None:
            # Create new child nodes based on the split
            node.feature = best_feature
            node.threshold = best_thresh
            node.value = None
            node.left = Node(value=node.majority_label)
            node.right = Node(value=node.majority_label)

    def _best_split(self, X, label_counts, feat_idxs):
        """
        Find the best split point among a subset of features.
        """
        best_gain = -1
        split_idx, split_threshold = None, None

        if isinstance(label_counts, Counter):
            labels = list(label_counts.elements())
        else:
            labels = list(label_counts)

        if not labels:
            return split_idx, split_threshold

        for feat_idx in feat_idxs:
            X_column = X[feat_idx]
            thresholds = [X_column]  # Single data point threshold

            for thr in thresholds:
                # Calculate information gain
                gain = self._information_gain(labels, [X_column], thr)

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

    def _most_common_label(self, labels):
        if isinstance(labels, Counter):
            counter = labels
        else:
            counter = Counter(labels)

        if not counter:
            return None

        return counter.most_common(1)[0][0]

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
            print(
                f"{indent}Leaf: Predict={node.value}, Stats={dict(node.split_stats)}"
            )
            return

        # Print current node's feature and threshold
        print(
            f"{indent}Node: Feature={node.feature}, Threshold={node.threshold}, Majority={node.majority_label}"
        )

        # Recurse into left and right subtrees if they exist
        if node.left:
            print(f"{indent}  Left:")
            self.print_tree(node.left, depth + 1)
        if node.right:
            print(f"{indent}  Right:")
            self.print_tree(node.right, depth + 1)

    def predict(
        self, X, *, label=None, node=None, depth=0, grace_period=200, delta=0.05
    ):
        """
        Predict a label for a single data point X, with Lazy Pruning.

        Args:
            X (array-like): The data point to predict.
            label (int, optional): Ground-truth label if available. When provided,
                lazy pruning statistics are updated using this label.
            node (Node): The current node to evaluate. Defaults to root.
            depth (int): The depth of the current node in the tree.
            grace_period (int): The minimum number of samples required to consider pruning.
            delta (float): Confidence parameter for Hoeffding Bound.

        Returns:
            int: Predicted label for the data point X.
        """
        if node is None:
            node = self.root  # Start prediction from the root

        if node is None:
            raise ValueError("The tree has not been initialized. Call `update` first.")

        # If the node is a leaf, return its prediction
        if node.is_leaf_node():
            return node.value if node.value is not None else node.majority_label

        predicted_label = node.majority_label

        if label is not None and predicted_label is not None:
            stats = node.pruning_stats
            stats["sample_count"] += 1
            if label != predicted_label:
                stats["misclassification_count"] += 1

            if stats["sample_count"] >= grace_period:
                misclassification_rate = (
                    stats["misclassification_count"] / stats["sample_count"]
                )
                epsilon = np.sqrt(np.log(1 / delta) / (2 * stats["sample_count"]))

                if misclassification_rate < epsilon:
                    # Collapse the branch into a leaf node using the stored majority label
                    node.left = None
                    node.right = None
                    node.feature = None
                    node.threshold = None
                    node.value = node.majority_label
                    node.pruning_stats = {"sample_count": 0, "misclassification_count": 0}
                    return node.value

        # Continue traversal based on the feature and threshold
        if node.feature is None or node.threshold is None:
            return predicted_label

        if X[node.feature] <= node.threshold:
            next_node = node.left
        else:
            next_node = node.right

        if next_node is None:
            return predicted_label

        return self.predict(
            X,
            label=label,
            node=next_node,
            depth=depth + 1,
            grace_period=grace_period,
            delta=delta,
        )
