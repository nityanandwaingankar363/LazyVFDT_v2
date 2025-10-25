import numpy as np
from collections import deque


class Node:
    id_counter = 0

    def __init__(
        self, feature=None, threshold=None, left=None, right=None, *, value=None
    ):
        self.id = Node.id_counter
        Node.id_counter += 1
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.stats = {"class_counts": {}, "total_samples": 0}

    def is_leaf(self):
        return self.value is not None

    def split(self, feature, threshold):
        print(f"Splitting Node {self.id} on Feature {feature} at Threshold {threshold}")
        self.feature = feature
        self.threshold = threshold
        self.left = Node()
        self.right = Node()

    def prune(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = max(self.stats["class_counts"], key=self.stats["class_counts"].get)


class SlidingVFDT:
    def __init__(self, delta=0.01, window_size=1000):
        """
        Initialize the CVFDT algorithm.
        :param delta: Confidence parameter for Hoeffding bound.
        :param window_size: Size of the sliding window.
        """
        self.delta = delta
        self.window_size = window_size
        self.tree = Node()  # Initialize with a root node
        self.sliding_window = deque(maxlen=window_size)

    # def predict(self, X):
    #     """
    #     Predict the label for a single instance.
    #     :param X: Feature array.
    #     :return: Predicted label.
    #     """
    #     current_node = self.tree
    #     while not current_node.is_leaf():
    #         feature = current_node.feature
    #         threshold = current_node.threshold
    #         if feature is None or threshold is None:
    #             break  # In case of incomplete split setup
    #         current_node = (
    #             current_node.left if X[feature] <= threshold else current_node.right
    #         )
    #     # Return the most frequent label if statistics are available
    #     if current_node.stats["class_counts"]:
    #         return max(
    #             current_node.stats["class_counts"],
    #             key=current_node.stats["class_counts"].get,
    #         )
    #     else:
    #         return None  # No prediction if stats are empty
    def predict(self, X):
        current_node = self.tree
        path = []
        while not current_node.is_leaf():
            path.append(
                f"Node {current_node.id}: Feature {current_node.feature}, Threshold {current_node.threshold}"
            )
            feature = current_node.feature
            threshold = current_node.threshold
            if feature is None or threshold is None:
                break
            current_node = (
                current_node.left if X[feature] <= threshold else current_node.right
            )
        print(f"Prediction Path: {' -> '.join(path)}")
        if current_node.stats["class_counts"]:
            return max(
                current_node.stats["class_counts"],
                key=current_node.stats["class_counts"].get,
            )
        else:
            return None

    def hoeffding_bound(self, n):
        """
        Calculate Hoeffding bound.
        :param n: Number of observations.
        :return: Hoeffding bound value.
        """
        return np.sqrt(np.log(1 / self.delta) / (2 * n))

    def update(self, X, y):
        """
        Update the decision tree with new instances.
        :param X: Features.
        :param y: Labels.
        """
        self.sliding_window.append((X, y))
        # Perform updates on the tree using the sliding window
        self._update_tree()

    def _update_tree(self):
        """
        Update the decision tree structure based on the current sliding window.
        This includes recalculating statistics, checking Hoeffding bounds,
        and adjusting nodes (splitting or pruning) dynamically.
        """
        # Track statistics for each node
        node_statistics = {}

        # Update node statistics with sliding window data
        for X, y in self.sliding_window:
            current_node = self.tree  # will always pick root node
            while not current_node.is_leaf():
                feature = current_node.feature
                threshold = current_node.threshold
                if feature is None or threshold is None:
                    break  # Avoid comparing None with numeric values
                current_node = (
                    current_node.left if X[feature] <= threshold else current_node.right
                )
            # Update statistics at the leaf node
            current_node.stats["total_samples"] += 1
            current_node.stats["class_counts"][y] = (
                current_node.stats["class_counts"].get(y, 0) + 1
            )
            # Track node statistics
            node_statistics[current_node.id] = current_node.stats

        # Adjust the tree based on Hoeffding bounds
        for node_id, stats in node_statistics.items():
            node = self._find_node_by_id(self.tree, node_id)  # Helper to find the node
            if node.is_leaf():
                # Skip processing for leaf nodes
                continue

            total_samples = stats["total_samples"]
            class_counts = stats["class_counts"]
            if total_samples < 2:
                # Not enough data to calculate meaningful splits
                continue

            # Sort class counts in descending order to get the two most frequent classes
            sorted_counts = sorted(
                class_counts.items(), key=lambda item: item[1], reverse=True
            )
            g1 = sorted_counts[0][1] if len(sorted_counts) > 0 else 0
            g2 = sorted_counts[1][1] if len(sorted_counts) > 1 else 0

            # Calculate the Hoeffding bound
            hoeffding_bound = self.hoeffding_bound(total_samples)

            # Check if the best split candidate is significantly better than the second best
            if (g1 - g2) > hoeffding_bound:
                # Identify the best feature and threshold to split
                best_feature, best_threshold = self._find_best_split(node, class_counts)

                # Perform the split if a valid feature and threshold are found
                if best_feature is not None and best_threshold is not None:
                    node.split(best_feature, best_threshold)
                    print(
                        f"Splitting Node {node.id}: Feature {best_feature}, Threshold {best_threshold}"
                    )

    def _find_node_by_id(self, node, node_id):
        """
        Recursively find a node by its ID in the decision tree.
        :param node: Current node to search.
        :param node_id: ID of the node to find.
        :return: Node with the given ID or None if not found.
        """
        if node is None:
            return None
        if node.id == node_id:
            return node
        left_result = self._find_node_by_id(node.left, node_id)
        if left_result:
            return left_result
        return self._find_node_by_id(node.right, node_id)

    def _find_best_split(self, node, class_counts):
        """
        Determine the best feature and threshold to split the given node.
        :param node: The node to evaluate.
        :param class_counts: Class distribution at this node.
        :return: (best_feature, best_threshold) or (None, None) if no split is found.
        """
        best_feature = None
        best_threshold = None
        best_gini = float("inf")  # Smaller Gini index is better

        # Total samples at this node
        total_samples = sum(class_counts.values())

        # Iterate over all features and potential split thresholds
        for feature in range(len(self.sliding_window[0][0])):  # Number of features
            thresholds = np.unique(
                [sample[feature] for sample, _ in self.sliding_window]
            )  # "sample, _ " is "X, y"
            for threshold in thresholds:
                # Split samples based on the threshold
                left_classes = {}
                right_classes = {}
                for sample, label in self.sliding_window:
                    if sample[feature] <= threshold:
                        left_classes[label] = left_classes.get(label, 0) + 1
                    else:
                        right_classes[label] = right_classes.get(label, 0) + 1

                left_total = sum(left_classes.values())
                right_total = sum(right_classes.values())

                if left_total == 0 or right_total == 0:
                    continue  # Avoid invalid splits

                # Calculate Gini index for the split
                gini_left = 1 - sum(
                    (count / left_total) ** 2 for count in left_classes.values()
                )
                gini_right = 1 - sum(
                    (count / right_total) ** 2 for count in right_classes.values()
                )
                weighted_gini = (left_total / total_samples) * gini_left + (
                    right_total / total_samples
                ) * gini_right

                # Update the best split if this is better
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature
                    best_threshold = threshold
        print(
            f"Evaluating Feature {feature}, Threshold {threshold}, Gini: {weighted_gini}"
        )

        return best_feature, best_threshold

    def print_tree(self, node, depth=0):
        prefix = "  " * depth
        if node.is_leaf():
            print(f"{prefix}Leaf Node: Value={node.value}, Stats={node.stats}")
        else:
            print(
                f"{prefix}Node ID: {node.id}, Feature={node.feature}, Threshold={node.threshold}"
            )
            self.print_tree(node.left, depth + 1)
            self.print_tree(node.right, depth + 1)

    # Use this after training
    # print_tree(self.tree)


###### NOte TO SELF
# threshold assignment is not working properly
