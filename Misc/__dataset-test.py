from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
monk_s_problems = fetch_ucirepo(id=70) 
  
# data (as pandas dataframes) 
X = monk_s_problems.data.features 
Y = monk_s_problems.data.targets 

import numpy as np

print("Updating tree with streaming data...\n")

tree = StreamingDecisionTree(min_samples_split=2, max_depth=3, grace_period=2, n_features=2)
# Initialize arrays
new_X = np.empty((0, X.shape[1]))  # Initialize empty 2D array with correct column size
new_Y = np.array([])  # Initialize empty 1D array
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
    new_X, new_Y, test_size=0.2, random_state=42, stratify=Y
)

# Print the shapes of the resulting datasets
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Y_train shape: {Y_train.shape}")
print(f"Y_test shape: {Y_test.shape}")

## START TRAINING THE TREE ##############\

print("Updating tree with streaming data...\n")

for i in range(len(X_train)):
    tree.update(X_train[i], Y_train[i])


#STARTE PREDICTION ###############
for i in range(len(X_test)):
    prediction = tree.predict(X_test[i])
    print(f"Data Point {i+1}: X={X_test[i]} => Prediction: {Y_test[i]}")

