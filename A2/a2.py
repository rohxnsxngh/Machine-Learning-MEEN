import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import os

# Specify the directory containing the CSV file
data_directory = 'C:\_dev\MEEN-423\A2'

# Change the working directory to the specified directory
os.chdir(data_directory)

# Load Training Data
raw_data = pd.read_csv("train_dataset.csv")
print(raw_data)