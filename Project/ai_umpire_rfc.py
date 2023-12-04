import pandas as pd
import matplotlib.pyplot as plt	
import matplotlib.patches as patch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import os

# Specify the directory containing the CSV file
#data_directory = 'C:\_dev\MEEN-423\Project'

# # Change the working directory to the specified directory
#os.chdir(data_directory)

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
    
    # ALL PITCHES PLOT #
    plt.figure(figsize=(6, 8))
        
    plt.scatter(umpire_df['px'][umpire_df['pcall'] == 0], umpire_df['pz'][umpire_df['pcall'] == 0], color = 'blue', label = 'Ball')
    plt.scatter(umpire_df['px'][umpire_df['pcall'] == 1], umpire_df['pz'][umpire_df['pcall'] == 1], color = 'red', label = 'Strike')
    
    zone = patch.Rectangle((px_left, sz_bot_avg), (px_right - px_left), (sz_top_avg - sz_bot_avg), fill=False)
    plt.gca().add_patch(zone)
    
    plt.title(f'All Pitches - {HP}')
    plt.xlabel('X Position (ft)')
    plt.ylabel('Z Position (ft)')
    plt.xlim(-3, 3)
    plt.ylim(-1, 6)
    plt.legend(loc ='upper right')
    #plt.savefig(f'All Pitches - {HP}.png')
    plt.show()

    # TEST PITCHES PLOT #    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=None)

    rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=10, min_samples_leaf=1)
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)    

    x0_values = np.linspace(-3, 3, 500)
    x1_values = np.linspace(-1, 6, 500)
    x0, x1 = np.meshgrid(x0_values, x1_values)
    X_contor = np.c_[x0.ravel(), x1.ravel()]

    y_contor = rf_model.predict(X_contor)

    contor_values = y_contor.reshape(x0.shape)
    contour_colors = [(0.5, 0.5, 1), (1, 0.5, 0.5)]

    plt.figure(figsize=(6, 8))
    
    plt.contourf(x0, x1, contor_values, levels=[0, 0.5, 1], colors=contour_colors, alpha=0.3)
     
    plt.scatter(X_test['px'][y_test==0], X_test['pz'][y_test==0], color = 'blue', label = 'Ball')
    plt.scatter(X_test['px'][y_test==1], X_test['pz'][y_test==1], color = 'red', label = 'Strike')

    plt.legend()
    
    zone = patch.Rectangle((px_left, sz_bot_avg), (px_right - px_left), (sz_top_avg - sz_bot_avg), fill=False)
    plt.gca().add_patch(zone)
    
    plt.text(-2.82, 5.71, f'Accuracy: {accuracy:.2f}', bbox = dict(facecolor='white', edgecolor='white', boxstyle='round'))
    
    plt.title(f'Random Forest Classifier - {HP}')
    plt.xlabel('X Position (ft)')
    plt.ylabel('Z Position (ft)')
    plt.xlim(-3, 3)
    plt.ylim(-1, 6)
    plt.legend(loc ='upper right')
    #plt.savefig(f'Random Forest Classifier - {HP}.png')
    plt.show()
