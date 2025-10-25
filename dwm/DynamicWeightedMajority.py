from collections import deque
import random
import numpy as np
import pandas as pd
from lazy_decision_tree import LazyDecisionTree
from sklearn.model_selection import ParameterGrid
from sklearn.tree import DecisionTreeClassifier


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
        # x = np.array(x).reshape(1, -1)  # Ensure input is 2D
        self.classifier.fit(x, y)


from collections import deque
import numpy as np
import random


class DynamicWeightedMajority:
    def __init__(
        self, num_classes, beta, theta, p, create_classifier, num_features, window_size
    ):
        """
        Initialize the DWM algorithm with a sliding window.
        :param num_classes: Number of classes (e.g., 2 for binary classification).
        :param beta: Factor for decreasing weights, 0 ≤ β < 1.
        :param theta: Threshold for deleting experts.
        :param p: Period between expert updates.
        :param create_classifier: Function to create a new base learner for each expert.
        :param num_features: Number of features in the input data.
        :param window_size: Size of the sliding window.
        """
        self.num_classes = num_classes
        self.beta = beta
        self.theta = theta
        self.p = p
        self.num_features = num_features
        self.create_classifier = create_classifier
        self.experts = []
        self.weights = []
        self.window_size = window_size

        # Initialize sliding window
        self.window_X = deque(maxlen=window_size)
        self.window_y = deque(maxlen=window_size)

        # Initialize the first expert with meaningful dummy data
        dummy_x = np.zeros((10, num_features))  # 10 samples of zeroed features
        dummy_y = np.zeros(10)  # 10 samples with label 0
        initial_expert = Expert(create_classifier)
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
        for weight, expert in zip(self.weights, self.experts):
            prediction = expert.classify(x)
            sigma[int(prediction)] += weight
        return np.argmax(sigma)

    def update(self, x, y, iteration):
        """
        Update the DWM model with a new example.
        :param x: Feature vector.
        :param y: True label.
        :param iteration: Current iteration number.
        """
        # Add the new sample to the sliding window
        self.window_X.append(x)
        self.window_y.append(y)

        # Step 1: Compute weighted predictions and update expert weights
        sigma = np.zeros(self.num_classes)
        for idx, (weight, expert) in enumerate(zip(self.weights, self.experts)):
            prediction = expert.classify(x)
            if iteration % self.p == 0 and prediction != y:
                # Decrease weight for incorrect predictions
                self.weights[idx] *= self.beta
            sigma[int(prediction)] += weight

        # Step 2: Make a global prediction
        global_prediction = np.argmax(sigma)

        # Step 3: Periodic expert maintenance
        if iteration % self.p == 0:
            # Normalize weights
            total_weight = sum(self.weights)
            if total_weight > 0:
                self.weights = [w / total_weight for w in self.weights]

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
                new_expert = Expert(self.create_classifier)
                # Train the new expert on the entire dataset seen so far
                new_expert.train(np.array(self.window_X), np.array(self.window_y))
                self.experts.append(new_expert)
                self.weights.append(1.0)

        # Step 4: Retrain all experts on the entire dataset seen so far
        for expert in self.experts:
            expert.train(np.array(self.window_X), np.array(self.window_y))

        # Step 4: Retrain all experts on the sliding window
        # for expert in self.experts:
        #     for xi, yi in zip(self.window_X, self.window_y):
        #         expert.train(xi, yi)
