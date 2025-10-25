from lazy_decision_tree import StreamingDecisionTree
from sklearn.model_selection import ParameterGrid
import numpy as np

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
    new_Y = np.append(new_Y, y2[0])  # Append the target value to the array

# Split the dataset into training and testing sets using scikit-learn
from sklearn.model_selection import train_test_split

# Perform a stratified split to maintain class balance in training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    new_X, new_Y, test_size=0.3, random_state=42, stratify=Y
)
# Perform hyperparameter tuning using grid search

# Define the parameter grid for the streaming decision tree
# param_grid = {
#     "min_samples_split": [1, 3, 5],
#     "max_depth": [954, 946],
#     "grace_period": [3, 2],
#     "n_features": [6, 5, 4, 3],
# }

param_grid = {
    "min_samples_split": [_ for _ in range(1, 51, 5)],
    "max_depth": [_ for _ in range(1, 1000, 5)],
    "grace_period": [_ for _ in range(1, 21, 2)],
    "n_features": [1, 2, 3, 4, 5, 6, X.shape[1]],
}

# Initialize variables to track the best parameters and accuracy
best_accuracy = 0
best_params = None

# Iterate over all parameter combinations in the grid
for params in ParameterGrid(param_grid):
    # Create a new decision tree with the current parameters
    tree = StreamingDecisionTree(**params)

    # Update the tree with the training data
    for i in range(len(X_train)):
        tree.update(X_train[i], Y_train[i])

    # tree.print_tree()
    # print("************\n")
    # Evaluate the tree's accuracy on the test set
    correct_predictions = 0
    total_predictions = len(X_test)

    for i in range(total_predictions):
        prediction = tree.predict(X_test[i])  # Get the prediction for the test sample
        if prediction == Y_test[i]:  # Compare with the actual label
            correct_predictions += 1

    # Calculate accuracy for the current parameter combination
    accuracy = correct_predictions / total_predictions

    # Update the best parameters and accuracy if the current accuracy is higher
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = params

# tree.print_tree()

# Print the best parameters and the corresponding accuracy
print(f"Best Parameters: {best_params}\n Best Accuracy: {best_accuracy * 100:.2f}%")

tree = StreamingDecisionTree(**best_params)
# Initialize the StreamingDecisionTree
# tree = StreamingDecisionTree(
#     min_samples_split=2,
#     max_depth=10,
#     grace_period=50,
#     n_features=2,
#     decay_rate=0.99,
#     delta=0.05,
# )

# Simulated streaming data (X: features, y: labels)
# np.random.seed(42)
# stream_size = 500  # Number of data points in the stream
# feature_count = 2  # Number of features

# Generate a simple linearly separable dataset for demonstration
# X_stream = np.random.rand(stream_size, feature_count)
# y_stream = (X_stream[:, 0] + X_stream[:, 1] > 1).astype(
#     int
# )  # Label: 1 if sum of features > 1, else 0

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
