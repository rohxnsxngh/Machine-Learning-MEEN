import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from joblib import dump
import os

# Specify the directory containing the CSV file
data_directory = 'C:\_dev\MEEN-423\A2'

# Change the working directory to the specified directory
os.chdir(data_directory)

# Load Training Data
data = pd.read_csv("train_dataset.csv")
# print(data.head())

# Define the features and target variable
X = data[['Rotational speed (RPM)', 'Load on the bearing (Newton)', 'Hardness of the material (HB)']]
y = data['Wear rate']

# Part 1: Polynomial Models
poly_orders = list(range(1, 7))
avg_rmse_scores = []

for order in poly_orders:
    # Create a polynomial feature transformer
    poly = PolynomialFeatures(degree=order, include_bias=False)
    X_poly = poly.fit_transform(X)

    # Perform 5-fold cross-validation
    model = LinearRegression()
    rmse_scores = np.sqrt(np.absolute(-cross_val_score(model, X_poly, y, cv=5, scoring="neg_mean_squared_error")))
    avg_rmse = np.mean(rmse_scores)
    # print("rsme", avg_rmse)
    avg_rmse_scores.append(avg_rmse)

# Determine the best polynomial order
best_poly_order = poly_orders[np.argmin(avg_rmse_scores)]
print("Best Polynomial Order:", best_poly_order)

# Retrain the best model on the entire training dataset
poly = PolynomialFeatures(degree=best_poly_order)
X_poly = poly.fit_transform(X)
best_model = LinearRegression()
best_model.fit(X_poly, y)

# Save the best model using joblib
dump(best_model, 'best_polynomial_model.joblib')

# Part 2: Ridge-regularized Polynomial Models
alpha_values = [0.001, 0.01, 0.1, 1]
best_avg_rmse = float('inf')
best_poly_order_ridge = None
best_alpha_ridge = None

for order in poly_orders:
    for alpha in alpha_values:
        # Create a polynomial feature transformer
        poly = PolynomialFeatures(degree=order, include_bias=False)
        X_poly = poly.fit_transform(X)

        # Standardize features
        scaler = StandardScaler()
        X_poly_scaled = scaler.fit_transform(X_poly)

        # Perform 5-fold cross-validation with Ridge regression
        model = Ridge(alpha=alpha)
        rmse_scores = np.sqrt(np.absolute(-cross_val_score(model, X_poly_scaled, y, cv=5, scoring="neg_mean_squared_error")))
        avg_rmse = np.mean(rmse_scores)
        # print("rmse scores", avg_rmse)

        if avg_rmse < best_avg_rmse:
            best_avg_rmse = avg_rmse
            best_poly_order_ridge = order
            best_alpha_ridge = alpha

print("Best Polynomial Order (Ridge):", best_poly_order_ridge)
print("Best Alpha (Ridge):", best_alpha_ridge)

# Retrain the best Ridge-regularized model on the entire training dataset
poly = PolynomialFeatures(degree=best_poly_order_ridge)
X_poly = poly.fit_transform(X)
scaler = StandardScaler()
X_poly_scaled = scaler.fit_transform(X_poly)
best_model_ridge = Ridge(alpha=best_alpha_ridge)
best_model_ridge.fit(X_poly_scaled, y)

# Save the best Ridge-regularized model using joblib
dump(best_model_ridge, 'best_ridge_model.joblib')