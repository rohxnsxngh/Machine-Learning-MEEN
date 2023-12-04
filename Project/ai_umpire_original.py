import pandas as pd
import matplotlib.pyplot as plt	
import matplotlib.patches as patch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


# Load the dataset
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


#%%
'''
plt.figure().set_figwidth(6)
plt.figure().set_figheight(8)

for i in range(len(gameHP)):
    if gameHP[i] == 'Malachi Moore' and pcall[i] == 'Ball':
        plt.scatter(px[i], pz_norm[i], color='blue')
    elif gameHP[i] == 'Malachi Moore' and pcall[i] == 'Strike':
        plt.scatter(px[i], pz_norm[i], color='red')

zone = patch.Rectangle((px_left, sz_bot_avg), (px_right - px_left), (sz_top_avg - sz_bot_avg), fill=False)
plt.gca().add_patch(zone)
plt.xlim(-3, 3)
plt.ylim(-1, 6)
plt.show()
'''

#%%

px_moore = []
pz_moore = []
pcall_moore = []

for i in range(len(gameHP)):
    if gameHP[i] == 'Angel Hernandez':
        px_moore.append(px[i])
        pz_moore.append(pz_norm[i])
        if pcall[i] == 'Strike':
            pcall_moore.append(1)
        else:
            pcall_moore.append(0)

moore = {
  'px': px_moore,
  'pz': pz_moore,
  'pcall': pcall_moore
}

#load data into a DataFrame object:
df_moore = pd.DataFrame(moore)

X = df_moore[['px', 'pz']]
y = df_moore['pcall']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.025, random_state=None)


# Define a parameter grid to search over
param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create a GridSearchCV object to find the best hyperparameters
grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Create and train a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=5)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

#create lists for plotting
x0s = np.linspace(-3, 3, 500)
x1s = np.linspace(-1, 6, 500)
x0, x1 = np.meshgrid(x0s, x1s)
X_new = np.c_[x0.ravel(), x1.ravel()]

#create loop for coefficents
    
#predict class labels
y_pred = rf_model.predict(X_new)

zz = y_pred.reshape(x0.shape)
contour_colors = [(0.5, 0.5, 1), (1, 0.5, 0.5)]





#create plot
plt.figure().set_figwidth(6)
plt.figure().set_figheight(8)

plt.contourf(x0, x1, zz, levels=[0, 0.5, 1], colors=contour_colors, alpha=0.3)
 
plt.scatter(X_test['px'][y_test==0], X_test['pz'][y_test==0], color = 'blue', label = 'Ball')
plt.scatter(X_test['px'][y_test==1], X_test['pz'][y_test==1], color = 'red', label = 'Strike')

plt.legend()

#plt.legend()
zone = patch.Rectangle((px_left, sz_bot_avg), (px_right - px_left), (sz_top_avg - sz_bot_avg), fill=False)
plt.gca().add_patch(zone)
plt.xlim(-3, 3)
plt.ylim(-1, 6)

#new_data = pd.DataFrame({'x_position': [X_new], 'z_position': [Z_new]})
#prediction = model.predict(new_data)