import numpy as np
import random
from sklearn.tree import DecisionTreeClassifier

# lambda: DecisionTreeClassifier(max_depth=3)
from sklearn.naive_bayes import GaussianNB

# create_classifier=lambda: GaussianNB()
from sklearn.neighbors import KNeighborsClassifier

# create_classifier=lambda: KNeighborsClassifier(n_neighbors=5)


from sklearn.ensemble import RandomForestClassifier

# create_classifier = lambda: RandomForestClassifier(n_estimators=10, max_depth=5)

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid


class Expert:
    """
    Represents an individual expert in the DWM algorithm.
    Each expert can be modeled using a base learner (e.g., a simple classifier).
    """

    def __init__(self, create_classifier):
        self.classifier = create_classifier()
        # print(f"Expert classifier is {self.classifier}")

    def classify(self, x):
        """Classify an input using the expert's classifier."""
        x = np.array(x).reshape(1, -1)  # Ensure input is 2D
        return self.classifier.predict(x)[0]

    def train(self, x, y):
        """Update the expert's classifier with a new training example."""
        x = np.array(x).reshape(1, -1)  # Ensure input is 2D
        self.classifier.fit(x, [y])


class DynamicWeightedMajority:
    def __init__(self, num_classes, beta, theta, p, create_classifier, num_features):
        """
        Initialize the DWM algorithm.
        :param num_classes: Number of classes (e.g., 2 for binary classification).
        :param beta: Factor for decreasing weights, 0 ≤ β < 1.
        :param theta: Threshold for deleting experts.
        :param p: Period between expert updates.
        :param create_classifier: Function to create a new base learner for each expert.
        :param num_features: Number of features in the input data.
        """
        self.num_classes = num_classes
        self.beta = beta
        self.theta = theta
        self.p = p
        self.num_features = num_features
        self.create_classifier = create_classifier
        self.experts = []
        self.weights = []

        # Initialize the first expert with dummy training data
        initial_expert = Expert(create_classifier)
        dummy_x = np.zeros(
            num_features
        )  # Create a feature vector with the correct dimension
        dummy_y = 1  # Initial label (e.g., for binary classification)
        initial_expert.train(dummy_x, dummy_y)
        self.experts.append(initial_expert)
        self.weights.append(1.0)

    def predict(self, x):
        """
        Make a prediction by aggregating the weighted votes of all experts.
        :param x: Feature vector.
        :return: Predicted class label.
        """
        sigma = np.zeros(self.num_classes)
        # print("Actual y is ", y)
        print("Input is ", x)
        for idx, (weight, expert) in enumerate(zip(self.weights, self.experts)):
            prediction = expert.classify(x)
            print(
                "For IDX", idx, "prediction is ", prediction, "and weight is ", weight
            )
            sigma[int(prediction)] += self.weights[idx]

        # Step 2: Make a global prediction
        print(f"sigma is {sigma}\n #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#")
        # global_prediction = np.argmax(sigma)
        return np.argmax(sigma)

    def update(self, x, y, iteration):
        """
        Update the DWM model with a new example.
        :param x: Feature vector.
        :param y: True label.
        :param iteration: Current iteration number.
        """
        # Step 1: Compute weighted predictions and update expert weights
        sigma = np.zeros(self.num_classes)
        print("#############################################")
        print("Actual y is ", y)
        for idx, (weight, expert) in enumerate(zip(self.weights, self.experts)):
            prediction = expert.classify(x)
            # print("prediction is ", prediction)
            print(f"prediction is {prediction} for index {idx}")
            if iteration % self.p == 0 and prediction != y:  #
                # Decrease weight for incorrect predictions
                self.weights[idx] *= self.beta
            sigma[int(prediction)] += self.weights[idx]

        # Step 2: Make a global prediction
        print(f"sigma is {sigma}")
        global_prediction = np.argmax(sigma)
        print(f"global prediction is {global_prediction}")
        # Step 3: Periodic expert maintenance
        if iteration % self.p == 0:
            print("'Periodic expert maintenance'")
            # Normalize weights
            total_weight = sum(self.weights)
            if total_weight > 0:

                self.weights = [w / total_weight for w in self.weights]
                print(f"Weights after normalization: {self.weights}")

            # Remove experts with weight below threshold
            remaining_experts = []
            remaining_weights = []
            for weight, expert in zip(self.weights, self.experts):
                if weight >= self.theta:
                    remaining_experts.append(expert)
                    remaining_weights.append(weight)
            self.experts = remaining_experts
            self.weights = remaining_weights

            # Add a new expert if the global prediction was incorrect
            if global_prediction != y:
                # Train the new expert on a small batch instead of a single sample
                batch_size = 5
                batch_indices = np.random.choice(
                    len(X_train), size=batch_size, replace=False
                )
                batch_X = X_train[batch_indices]
                batch_y = Y_train[batch_indices]

                new_expert = Expert(self.create_classifier)
                for xi, yi in zip(batch_X, batch_y):
                    new_expert.train(xi, yi)

                self.experts.append(new_expert)
                self.weights.append(1.0)

        print("printig weights and num of experts")
        print(self.weights, len(self.experts))  # Print the updated weights
        # Step 4: Train all experts on the new example
        for expert in self.experts:
            print(f"Training expert with data x {x} and y {y}")
            expert.train(x, y)


################################################################
# Importing the MONKS dataset using the UCIMLRepo library
from ucimlrepo import fetch_ucirepo

# Fetch dataset with ID 70 (MONKS dataset)
monk_s_problems = fetch_ucirepo(id=70)

# Separate dataset into features (X) and targets (Y)
X = monk_s_problems.data.features  # Feature data as a pandas DataFrame
Y = monk_s_problems.data.targets  # Target data as a pandas DataFrame

print("Updating tree with streaming data...\n")

# Initialize arrays to hold processed data
new_X = np.empty((0, X.shape[1]))  # Empty 2D array to store features
new_Y = np.array([])  # Empty 1D array to store target values

# Process the feature data row by row and convert to NumPy arrays
for x1, x2 in X.iterrows():
    new_X = np.vstack([new_X, x2.to_numpy()])  # Add new row to the feature array

# Process the target data row by row and convert to a 1D array
for y1, y2 in Y.iterrows():
    new_Y = np.append(new_Y, y2[0])
# print("unique y", len(np.unique(new_Y)))

X = new_X
y = new_Y
################################################################


X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

unique, counts = np.unique(y, return_counts=True)
print(f"Class Distribution in Training Data: {dict(zip(unique, counts))}")

param_grid = {
    "num_classes": [2],
    "beta": [0.2],  # Penalize incorrect predictions more strongly
    "theta": [0.23],  # Remove ineffective experts faster
    "p": [2],  # Update experts more frequently
    "create_classifier": [
        lambda: RandomForestClassifier(n_estimators=10, max_depth=5)
    ],  # Allow deeper trees
    "num_features": [X.shape[1]],
}

best_accuracy = 0
best_params = None


for params in ParameterGrid(param_grid):
    # Create a new decision tree with the current parameters
    dwm = DynamicWeightedMajority(**params)

    # Update the tree with the training data
    for i in range(len(X_train)):
        dwm.update(X_train[i], Y_train[i], i)

    correct_predictions = 0
    total_predictions = len(X_test)

    for i in range(total_predictions):
        prediction = dwm.predict(X_test[i])  # Get the prediction for the test sample
        if prediction == Y_test[i]:  # Compare with the actual label
            correct_predictions += 1

    # Calculate accuracy for the current parameter combination
    accuracy = correct_predictions / total_predictions

    # Update the best parameters and accuracy if the current accuracy is higher
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = params

# Print the best parameters and the corresponding accuracy
print(f"Best Parameters: {best_params}\n Best Accuracy: {best_accuracy * 100:.2f}%")


# maintain all input data and train using data seen so far. or a sliding window of inputs
