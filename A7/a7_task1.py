import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import os

# # Specify the directory containing the CSV file
# data_directory = 'C:\_dev\MEEN-423\A7'

# # # Change the working directory to the specified directory
# os.chdir(data_directory)

# Load the dataset
data = pd.read_csv('gt_2014.csv')  # Replace 'your_dataset.csv' with the actual filename

# Extract features and response variable
X = data.drop('NOX', axis=1)  # Assuming 'NOX' is the column name for the response variable
y = data['NOX']

# Standardize the features
X_standardized = (X - X.mean()) / X.std()

# Perform PCA
pca = PCA()
pca.fit(X_standardized)

# Scree plot
plt.subplot(1, 2, 1)
plt.plot(np.arange(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')

# Explained variance plot
plt.subplot(1, 2, 2)
plt.plot(np.arange(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.title('Explained Variance Plot')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Proportion of Variance Explained')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()
