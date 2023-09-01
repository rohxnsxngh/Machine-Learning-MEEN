import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import os

# Specify the directory containing the CSV file
data_directory = 'C:\_dev\MEEN-423\A1'

# Change the working directory to the specified directory
os.chdir(data_directory)

# Load Training Data
raw_data = pd.read_csv("a1_data.csv")

# Seperate Input and Output Data
x = raw_data[['x1','x2']].to_numpy()
y = raw_data['y'].to_numpy()


# Divide Training and Test Sets
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 2022)

#Train the Data
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train) # train model for linear regression based on training data

y_train_pred = lin_reg.predict(x_train) # predict responses based on x_train
y_test_pred = lin_reg.predict(x_test) # predict responses based on x_test

linear_train_mse = mean_squared_error(y_train, y_train_pred)
linear_test_mse = mean_squared_error(y_test, y_test_pred)


# Do the same but for polynomial, degree 2
poly2_reg = LinearRegression()
poly2 = PolynomialFeatures(degree = 2, include_bias = False) # Set up polynomial, degree 2
poly2_x_train = poly2.fit_transform(x_train)
poly2_x_test = poly2.fit_transform(x_test)

poly2_reg.fit(poly2_x_train, y_train) # train model for poly, degree 2
poly2_train_pred = poly2_reg.predict(poly2_x_train) # Predict x train
poly2_test_pred = poly2_reg.predict(poly2_x_test) # Predict x test

poly2_train_mse = mean_squared_error(y_train, poly2_train_pred) # Find mse
poly2_test_mse = mean_squared_error(y_test, poly2_test_pred)

# Repeat for polynomial, degree 3
poly3_reg = LinearRegression()
poly3 = PolynomialFeatures(degree = 3, include_bias = False) # Set up polynomial, degree 3
poly3_x_train = poly3.fit_transform(x_train)
poly3_x_test = poly3.fit_transform(x_test)


poly3_reg.fit(poly3_x_train, y_train) # Train model
poly3_train_pred = poly3_reg.predict(poly3_x_train) # Predict x train
poly3_test_pred = poly3_reg.predict(poly3_x_test) # Predict x test

poly3_train_mse = mean_squared_error(y_train, poly3_train_pred) # Find mse
poly3_test_mse = mean_squared_error(y_test, poly3_test_pred)

def display_results_table(linear, quadratic, cubic):
  print('\n\t\t\tTrain MSE\tTest MSE')
  print('Linear    \t%6.4f\t\t%6.4f' % linear)
  print('Quadratic \t%6.4f\t\t%6.4f' % quadratic)
  print('Cubic     \t%6.4f\t\t%6.4f' % cubic)
  
linear = (linear_train_mse, linear_test_mse)
quadratic = (poly2_train_mse, poly2_train_mse) 
cubic = (poly3_train_mse, poly3_train_mse)

display_results_table(linear, quadratic, cubic)
  
  
  