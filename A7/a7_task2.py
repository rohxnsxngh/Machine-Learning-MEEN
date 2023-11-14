import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

# Specify the directory containing the CSV file
data_directory = 'C:\_dev\MEEN-423\A7'

# # Change the working directory to the specified directory
os.chdir(data_directory)

# Load the dataset
data = pd.read_csv('gt_2014.csv')  # Replace 'your_dataset.csv' with the actual filename

# Extract features and response variable
X = data.drop('NOX', axis=1)  # Assuming 'NOX' is the column name for the response variable
y = data['NOX']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
X_train_standardized = (X_train - X_train.mean()) / X_train.std()
X_test_standardized = (X_test - X_train.mean()) / X_train.std()  # Use mean and std from training set for test set

# Perform PCA based on the number of principal components decided in Task 1
n_components =  8 # number of principal components -  the eigenvalues start to flatten out indicates the number of principal components to keep
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_standardized)
X_test_pca = pca.transform(X_test_standardized)

# Build the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss='mean_squared_error')

# Train the model
history = model.fit(X_train_pca, y_train, epochs=100, validation_data=(X_test_pca, y_test), verbose=0)

# Evaluate the model
train_predictions = model.predict(X_train_pca).flatten()
test_predictions = model.predict(X_test_pca).flatten()

train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
r2 = r2_score(y_test, test_predictions)

# Print results
print(f'Training RMSE: {train_rmse}')
print(f'Test RMSE: {test_rmse}')
print(f'R^2 Score: {r2}')

# Plot training and test MSE vs epoch
plt.plot(history.history['loss'], label='Training MSE')
plt.plot(history.history['val_loss'], label='Test MSE')
plt.title('Training and Test MSE vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()
