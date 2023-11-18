import pandas as pd
import matplotlib.pyplot as plt	
import matplotlib.patches as patch
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import os

# Specify the directory containing the CSV file
data_directory = 'C:\_dev\MEEN-423\Project'

# # Change the working directory to the specified directory
os.chdir(data_directory)

data = pd.read_csv('ai_umpire_data.csv')

px = data['pX']
pz = data['pZ']
pcall = data['pCall']
sz_top = data['sZ_top']
sz_bot = data['sZ_bot']
gameHP = data['gameHP']

px_right = 0.83
px_left = -0.83

sz_top_avg = 3.5
sz_bot_avg = 1.5

RAD = 0.12
MOE = 0.08

pz_norm = []

for p in range(len(pz)):
    pz_prop = (pz[p] - sz_bot[p]) / (sz_top[p] - sz_bot[p])
    pz_norm.append((pz_prop * (sz_top_avg - sz_bot_avg)) + sz_bot_avg)

umpires = ['Angel Hernandez', 'Erich Bacchus', 'Junior Valentine', 'Malachi Moore', 'Pat Hoberg', 'Quinn Wolcott']

for HP in umpires:
    px_umpire = []
    pz_umpire = []
    pcall_umpire = []

    for n in range(len(gameHP)):
        if gameHP[n] == HP:
            px_umpire.append(px[n])
            pz_umpire.append(pz_norm[n])
            
            if pcall[n] == 'Strike':
                pcall_umpire.append(1)
            elif pcall[n] == 'Ball':
                pcall_umpire.append(0)

    umpire_data = {
      'px': px_umpire,
      'pz': pz_umpire,
      'pcall': pcall_umpire}

    umpire_df = pd.DataFrame(umpire_data)

    X = umpire_df[['px', 'pz']]
    y = umpire_df['pcall']
  
    # TEST PITCHES PLOT #
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=None)

    # Define the parameter grid to search
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5]
    }

    # Create the RandomForestClassifier
    rf_model = RandomForestClassifier()

    # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)

    # Perform the grid search
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters
    print(f"Best Hyperparameters for {HP}: {grid_search.best_params_}")

    # Access the best model directly
    best_rf_model = grid_search.best_estimator_
    
    # Evaluate the model on the test set
    y_pred = best_rf_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print("accuracy: ", accuracy)
