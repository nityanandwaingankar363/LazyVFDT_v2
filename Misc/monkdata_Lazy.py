from lazy_decision_tree import LazyDecisionTree
from sklearn.model_selection import ParameterGrid
import numpy as np
import matplotlib.pyplot as plt
import time
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split

# Fetch dataset with ID 70 (MONKS dataset)
monk_s_problems = fetch_ucirepo(id=70)

# Separate dataset into features (X) and targets (Y)
X = monk_s_problems.data.features  # Feature data as a pandas DataFrame
Y = monk_s_problems.data.targets  # Target data as a pandas DataFrame

print("Updating tree with streaming data...\n")

# Convert dataset to NumPy arrays
new_X = X.to_numpy()
new_Y = Y.to_numpy().flatten()

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    new_X, new_Y, test_size=0.3, random_state=42, stratify=new_Y
)

# # Define hyperparameters for grid search
# param_grid = {
#     "min_samples_split": [1],
#     "max_depth": [41],
#     "grace_period": [1],
#     "n_features": [4],
#     "seed": [6],
# }


accuracy_over_time = []  # Track accuracy over streamed records
processing_times = []

# for params in ParameterGrid(param_grid):
tree = LazyDecisionTree(min_samples_split=1, max_depth=41, grace_period=1, n_features=4)
train_start_time = time.time()

# Update the tree with the training data
for i in range(len(X_train)):
    tree.update(X_train[i], Y_train[i])

train_end_time = time.time()
training_time = train_end_time - train_start_time

# Evaluate the tree's accuracy on the test set
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

    # # Calculate accuracy for the current parameter combination
    accuracy = correct_predictions / total_predictions
    # if accuracy > best_accuracy:
    #     best_accuracy = accuracy
    #     best_params = params

# Plot accuracy over time
plt.figure(figsize=(10, 5))
plt.plot(
    range(len(accuracy_over_time)),
    accuracy_over_time,
    label="Lazy DT Accuracy",
    color="blue",
)
plt.xlabel("Number of Streamed Records")
plt.ylabel("Accuracy")
plt.title("Accuracy Over Time (Lazy Decision Tree on MONK Dataset)")
plt.legend()
plt.grid(True)
plt.show()

# Print final results
# print(f"\nBest Parameters: {best_params}")
print(f"Final Accuracy: {accuracy * 100:.2f}%")
print(f"Training Time: {training_time:.7f} seconds")
print(f"Average predicting Time per Record: {np.mean(processing_times):.7f} ms")
