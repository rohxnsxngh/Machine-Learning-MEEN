import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os


# Specify the directory containing the CSV file
data_directory = 'C:\_dev\MEEN-423\A5'

# Change the working directory to the specified directory
os.chdir(data_directory)

# Load the dataset
data = pd.read_csv("steel_strength.csv")

# Split the data into training and test sets
X = data.iloc[:, 1:-3]  # Features (13 alloying elements)
y = data['tensile strength']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Define the hyperparameter grid for SVR and perform hyperparameter tuning using GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100, 250, 500, 1000],
    'gamma': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
}

svr = SVR(kernel='rbf', epsilon=50)
grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_svr = grid_search.best_estimator_

# Create a unity plot using the test subset of the data
y_pred = best_svr.predict(X_test)

# Compute RMSE on the training set, RMSE on the test set, and r-squared on the test set
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel("True Tensile Strength [MPa]")
plt.ylabel("Predicted Tensile Strength [MPa]")
plt.title("Unity Plot")
plt.show()

y_train_pred = best_svr.predict(X_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE on the training set:", rmse_train)
print("RMSE on the test set:", rmse_test)
print("R-squared on the test set:", r2)
