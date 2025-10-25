from sklearn.model_selection import ParameterGrid
import numpy as np
from collections import deque
from cvfdt import SlidingVFDT

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

# Initialize the streaming decision tree with initial parameters
# tree = StreamingDecisionTree(min_samples_split=34, max_depth=1000, grace_period=6, n_features=6)

# Process the feature data row by row and convert to NumPy arrays
for x1, x2 in X.iterrows():
    new_X = np.vstack([new_X, x2.to_numpy()])  # Add new row to the feature array
    # print(f"Data Point type: {type(x2.to_numpy())}")
    # print(f"Data Point: {x2.to_numpy()}")
    # print(f"new_X:\n{new_X}\n")

# Process the target data row by row and convert to a 1D array
for y1, y2 in Y.iterrows():
    new_Y = np.append(new_Y, y2[0])  # Append the target value to the array
    # print(f"Data Point type: {type(y2[0])}")
    # print(f"Data Point: {y2[0]}")
    # print(f"new_Y:\n{new_Y}\n")
# print(f"new_X shape: {new_X.shape}, new_Y shape: {new_Y.shape}")
# print(f"Sample Features: {new_X[10:15]}, Sample Labels: {new_Y}")


# Split the dataset into training and testing sets using scikit-learn
from sklearn.model_selection import train_test_split

# Perform a stratified split to maintain class balance in training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    new_X, new_Y, test_size=0.3, random_state=42, stratify=Y
)
# print(X_train[:5], Y_train[:5])  # Check training data
# print(np.unique(Y_train, return_counts=True))  # Check label distribution


# Perform hyperparameter tuning using grid search
param_grid = {
    # "delta": [_ for _ in range(0, 1, 0.01)],
    "delta": [0.001, 0.05, 0.1],  # 0.12, 0.1, 0.01, 0.001],
    "window_size": [10, 50, 100],  # , 10, 100, 100],
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
        # print(f"Prediction at sample {i}: {prediction}, True Label: {Y_test[i]}")
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

tree = SlidingVFDT(**best_params)
# Stream the data one by one to the tree
print("Streaming data and training...")
for i in range(len(X_train)):
    tree.update(X_train[i], Y_train[i])

    # Optionally, predict periodically
    if i % 100 == 0 and i > 0:
        sample = X_train[i]
        prediction = tree.predict(sample)
        print(f"Sample {i}: True Label={Y_train[i]}, Predicted={prediction}")

# Predict on new unseen samples
print("\nMaking predictions on new data...")
# new_samples = np.random.rand(5, feature_count)
for i, sample in enumerate(X_test):
    prediction = tree.predict(sample)
    print(
        f"Sample {i}: Features={sample}, Predicted Label={prediction} : True Label={Y_test[i]}"
    )
