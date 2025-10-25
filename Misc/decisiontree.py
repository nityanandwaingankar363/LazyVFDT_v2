import numpy as np
from collections import deque
from ucimlrepo import (
    fetch_ucirepo,
)  # Importing the MONKS dataset using the UCIMLRepo library


class Node:
    def __init__(
        self, feature=None, threshold=None, left=None, right=None, *, value=None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


####################################################################################################


class VFDT:
    def __init__(self, delta=0.01, tau=0.05, grace_period=200, max_window_size=1000):
        self.delta = delta
        self.tau = tau
        self.grace_period = grace_period
        self.max_window_size = max_window_size
        self.window = deque(maxlen=max_window_size)
        self.tree = None

    def _gini(self, y):
        m = len(y)
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))

    def _best_split(self, X, y):
        m, n = X.shape
        if m <= 1:
            return None, None

        parent_gini = self._gini(y)
        best_gini = float("inf")
        best_idx, best_thr = None, None

        for idx in range(n):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = np.zeros(len(np.unique(y)))
            num_right = np.bincount(classes, minlength=len(np.unique(y)))

            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1

                gini_left = 1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(len(num_left))
                )
                gini_right = 1.0 - sum(
                    (num_right[x] / (m - i)) ** 2 for x in range(len(num_right))
                )
                gini = (i * gini_left + (m - i) * gini_right) / m

                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2

        return best_idx, best_thr

    def _grow_tree(self, X, y):
        num_samples_per_class = [np.sum(y == i) for i in np.unique(y)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(value=predicted_class)

        if len(set(y)) == 1:
            return node

        idx, thr = self._best_split(X, y)
        if idx is None:
            return node

        indices_left = X[:, idx] < thr
        X_left, y_left = X[indices_left], y[indices_left]
        X_right, y_right = X[~indices_left], y[~indices_left]

        node.feature = idx
        node.threshold = thr
        node.left = self._grow_tree(X_left, y_left)
        node.right = self._grow_tree(X_right, y_right)
        return node

    def fit(self, X, y):
        self.window.append((X, y))
        if len(self.window) >= self.grace_period:
            X_window = np.vstack([x for x, _ in self.window])
            y_window = np.hstack([y for _, y in self.window])
            self.tree = self._grow_tree(X_window, y_window)
            self.window.clear()

    def _predict(self, inputs):
        node = self.tree
        while not node.is_leaf_node():
            if inputs[node.feature] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict(self, X):
        return self._predict(X)


####################################################################################################
class CVFDT(VFDT):
    def __init__(self, delta=0.01, tau=0.05, grace_period=200, max_window_size=1000):
        super().__init__(delta, tau, grace_period, max_window_size)
        self.window = deque(maxlen=max_window_size)

    def _update_tree(self, X, y):
        # Update the tree with the new sample
        if self.tree is None:
            self.tree = self._grow_tree(X, y)
        else:
            self._update_node(self.tree, X, y)

    def _update_node(self, node, X, y):
        if node.is_leaf_node():
            return

        if X[node.feature] < node.threshold:
            self._update_node(node.left, X, y)
        else:
            self._update_node(node.right, X, y)

        # Check for potential split
        idx, thr = self._best_split(X, y)
        if idx is not None and (node.feature != idx or node.threshold != thr):
            node.feature = idx
            node.threshold = thr
            indices_left = X[:, idx] < thr
            X_left, y_left = X[indices_left], y[indices_left]
            X_right, y_right = X[~indices_left], y[~indices_left]
            node.left = self._grow_tree(X_left, y_left)
            node.right = self._grow_tree(X_right, y_right)

    def fit(self, X, y):
        self.window.append((X, y))
        self._update_tree(X, y)

        # Remove old samples from the window and update the tree
        if len(self.window) == self.max_window_size:
            old_X, old_y = self.window.popleft()
            self._update_tree(old_X, old_y)


if __name__ == "__main__":
    # Fetch dataset with ID 70 (MONKS dataset)
    monk_s_problems = fetch_ucirepo(id=70)

    # Separate dataset into features (X) and targets (Y)
    X = monk_s_problems.data.features  # Feature data as a pandas DataFrame
    Y = monk_s_problems.data.targets  # Target data as a pandas DataFrame

    # Initialize arrays to hold processed data
    new_X = np.empty((0, X.shape[1]))  # Empty 2D array to store features
    new_Y = np.array([])  # Empty 1D array to store target values

    # Process the feature data row by row and convert to NumPy arrays
    for x1, x2 in X.iterrows():
        new_X = np.vstack([new_X, x2.to_numpy()])  # Add new row to the feature array

    # Process the target data row by row and convert to a 1D array
    for y1, y2 in Y.iterrows():
        new_Y = np.append(new_Y, y2[0])  # Append the target value to the array

    # print("Printing new_X and new_Y")
    # print(new_X)
    # print(new_Y)
    # Split the dataset into training and testing sets using scikit-learn
    from sklearn.model_selection import train_test_split

    # Perform a stratified split to maintain class balance in training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(
        new_X, new_Y, test_size=0.3, random_state=50, stratify=new_Y
    )
    # print("Printing X_train and Y_train")
    # print(X_train)
    # print(Y_train)
    # Initialize CVFDT
    cvfdt = CVFDT(grace_period=50)

    # Simulate streaming data for training
    for i in range(len(X_train)):
        cvfdt.fit(
            X_train[i].reshape(1, -1),
            Y_train[i].reshape(
                1,
            ),
        )
        # time.sleep(0.01)  # Simulate delay in streaming data

    # Simulate streaming data for prediction
    predictions = []
    for i in range(len(X_test)):
        prediction = cvfdt.predict(X_test[i].reshape(1, -1))
        predictions.append(int(prediction))
        # time.sleep(0.01)  # Simulate delay in streaming data

    # Calculate accuracy
    accuracy = np.mean(predictions == Y_test)

    # Print predictions and accuracy
    print("Predictions:", predictions)
    print("Actual labels:", Y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")
