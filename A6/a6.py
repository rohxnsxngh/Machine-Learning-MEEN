import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# Step 1: Generate the moons dataset
X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)

# Step 2: Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Perform grid search for hyperparameters
param_grid = {'max_leaf_nodes': [None, 10, 20, 30, 40, 50]}
tree_classifier = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(tree_classifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_max_leaf_nodes = grid_search.best_params_['max_leaf_nodes']

# Step 4: Train the Decision Tree with the best hyperparameters
best_tree_classifier = DecisionTreeClassifier(max_leaf_nodes=best_max_leaf_nodes, random_state=42)
best_tree_classifier.fit(X_train, y_train)

# Step 5: Measure model performance on the test set
accuracy = best_tree_classifier.score(X_test, y_test)
print("Decision Tree Accuracy: {:.2f}%".format(accuracy * 100))


from sklearn.model_selection import ShuffleSplit
from scipy.stats import mode

# Step 1: Generate 1,000 subsets of the training set
n_trees = 1000
n_instances = 100
subsets = []

rs = ShuffleSplit(n_splits=n_trees, test_size=len(X_train) - n_instances, random_state=42)
for train_index, _ in rs.split(X_train):
    X_subset, y_subset = X_train[train_index], y_train[train_index]
    subsets.append((X_subset, y_subset))

# Step 2: Train Decision Trees on each subset
forest = [DecisionTreeClassifier(max_leaf_nodes=best_max_leaf_nodes, random_state=42).fit(X_subset, y_subset) for X_subset, y_subset in subsets]

# Step 3: Make predictions and find the majority vote
predictions = np.array([tree.predict(X_test) for tree in forest])
forest_predictions, _ = mode(predictions, axis=0)

# Step 4: Evaluate predictions on the test set
forest_accuracy = np.mean(forest_predictions == y_test)
print("Random Forest Accuracy: {:.2f}%".format(forest_accuracy * 100))
