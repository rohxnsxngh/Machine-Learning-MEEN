import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, auc, precision_recall_curve, confusion_matrix, classification_report
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Load the dataset (if not already loaded)
# Specify the directory containing the CSV file
data_directory = 'C:\_dev\MEEN-423\A4'

# Change the working directory to the specified directory
os.chdir(data_directory)
data = pd.read_csv("data.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 1. Data Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label='Normal', marker='o')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='Defective', marker='x')
plt.xlabel("Vibration Frequency (Hz)")
plt.ylabel("Temperature (°C)")
plt.title("Scatter Plot of Vibration Frequency vs. Temperature")
plt.legend()
plt.show()

# Observations:
# - The dataset appears to have a relatively balanced distribution between normal and defective components.

# 2. Data Splitting, Pre-processing and Model Training
# Split the data using the UIN-based random state
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=21)

# Standardize the data using only training data statistics
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Define the L2 regularization coefficients
l2_coeffs = [0, 0.01, 1]

# Initialize a dictionary to store the models and their histories
models = {}
histories = {}

# Loop over the L2 regularization coefficients
for coeff in l2_coeffs:
    # Train ANN with L2 regularization
    model = Sequential([
        Dense(20, activation='relu', kernel_regularizer=l2(coeff), input_shape=(X_train.shape[1],)),
        Dense(20, activation='relu', kernel_regularizer=l2(coeff)),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=2000, batch_size=32, verbose=0)

    # Store the model and its history
    models[coeff] = model
    histories[coeff] = history

# 3. Compute Accuracy
for coeff, model in models.items():
    y_pred = (model.predict(X_val) > 0.5).astype("int32").ravel()
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Model Accuracy with L2 regularization coefficient {coeff}: {accuracy}")

# Based on accuracy, which models are performing well and which are not?
# Accuracy can give a general idea of model performance, but it may not be sufficient to evaluate models, especially when dealing with imbalanced datasets.

# 4. Loss vs Epoch Visualization
for coeff, history in histories.items():
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss vs. Epoch with L2 regularization coefficient {coeff}")
    plt.legend()
    plt.show()

# Observations:
# - The training and validation losses generally decrease over epochs.
# - Models with lower regularization coefficients (e.g., 0) seem to converge faster.

# 5. Compute ROC-AUC
for coeff, model in models.items():
    y_prob = model.predict(X_val).ravel()
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) with L2 regularization coefficient {coeff}')
    plt.legend(loc='lower right')
    plt.show()

# How does the ROC-AUC score correlate with the model's ability to distinguish between the classes?
# - ROC-AUC measures the area under the ROC curve, which quantifies a model's ability to distinguish between classes.
# - A higher ROC-AUC score indicates better discrimination between classes.

# 6. Compute other metrics
for coeff, model in models.items():
    y_prob = model.predict(X_val).ravel()
    y_pred = (y_prob > 0.5).astype("int32")
    confusion = confusion_matrix(y_val, y_pred)
    report = classification_report(y_val, y_pred)

    print("Confusion Matrix:")
    print(confusion)
    print("\nClassification Report:")
    print(report)

# 7. Decision Boundary Visualization
x0s = np.linspace(X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5, 500)
x1s = np.linspace(X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5, 500)
x0, x1 = np.meshgrid(x0s, x1s)
X_new = np.c_[x0.ravel(), x1.ravel()]

for coeff, model in models.items():
    y_pred = model.predict(X_new)
    zz = y_pred.reshape(x0.shape)
    contour_colors = [(0.5, 0.5, 1), (1, 0.5, 0.5)]

    plt.figure(figsize=(10, 6))
    plt.contourf(x0, x1, zz, levels=[0, 0.5, 1], colors=contour_colors, alpha=0.3)
    
    plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], c="blue", label="Class 0")
    plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], c="red", label="Class 1", marker='s')    

    plt.xlabel("Vibration Frequency (Hz)")
    plt.ylabel("Temperature (°C)")
    plt.title(f"Decision Boundary Visualization with L2 regularization coefficient {coeff}")
    plt.legend()
    plt.show()

# 8. Model selection
# Based on the provided information, you can select the model based on the ROC-AUC score, classification report, and your specific requirements.

# write code to plot precision-recall curve
# Plot Precision-Recall curve for each model
for coeff, model in models.items():
    y_prob = model.predict(X_val).ravel()
    precision, recall, _ = precision_recall_curve(y_val, y_prob)
    
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label=f'Model with L2 regularization coefficient {coeff}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc='lower left')
    plt.show()

# 9. Threshold Selection
for coeff, model in models.items():
    y_prob = model.predict(X_val).ravel()
    precision, recall, thresholds = precision_recall_curve(y_val, y_prob)
    thresholds = np.append(thresholds, 1)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    plt.grid(True, which='both', axis='both', linestyle='--', color='gray')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Recall', color='blue')
    ax1.plot(thresholds, recall, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Precision', color='green')
    ax2.plot(thresholds, precision, color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    fig.tight_layout()
    plt.title(f'Recall and Precision as functions of Threshold with L2 regularization coefficient {coeff}')
    plt.show()

# 10. Final Evaluation
# Compute the chosen metric on the test data using the selected threshold value.

# Example: If you choose precision as the critical metric and set a threshold to achieve at least 0.8 precision:

# model 0
y_prob = models[0].predict(X_val).ravel()
precision, recall, thresholds = precision_recall_curve(y_val, y_prob)
threshold = thresholds[np.argmax(precision >= 0.8)]
y_test_pred = (model.predict(X_test) > threshold).astype("int32").ravel()
precision_test = precision_score(y_test, y_test_pred)
recall_test = recall_score(y_test, y_test_pred)

print(f"Precision on Test Data with L2 regularization coefficient {coeff}: {precision_test:.2f}")
print(f"Recall on Test Data with L2 regularization coefficient {coeff}: {recall_test:.2f}")

# model 1
y_prob = models[1].predict(X_val).ravel()
precision, recall, thresholds = precision_recall_curve(y_val, y_prob)
threshold = thresholds[np.argmax(precision >= 0.8)]
y_test_pred = (model.predict(X_test) > threshold).astype("int32").ravel()
precision_test = precision_score(y_test, y_test_pred)
recall_test = recall_score(y_test, y_test_pred)

print(f"Precision on Test Data with L2 regularization coefficient {coeff}: {precision_test:.2f}")
print(f"Recall on Test Data with L2 regularization coefficient {coeff}: {recall_test:.2f}")

# model 2
y_prob = models[2].predict(X_val).ravel()
precision, recall, thresholds = precision_recall_curve(y_val, y_prob)
threshold = thresholds[np.argmax(precision >= 0.8)]
y_test_pred = (model.predict(X_test) > threshold).astype("int32").ravel()
precision_test = precision_score(y_test, y_test_pred)
recall_test = recall_score(y_test, y_test_pred)

print(f"Precision on Test Data with L2 regularization coefficient {coeff}: {precision_test:.2f}")
print(f"Recall on Test Data with L2 regularization coefficient {coeff}: {recall_test:.2f}")

# Is the performance satisfactory? Discuss any potential improvements that could be made.

# The performance depends on the chosen metric (precision, recall, etc.) and the specific requirements of the problem.
# Additional improvements may include:
# - Fine-tuning hyperparameters (e.g., neural network architecture, regularization)
# - Handling class imbalance (e.g., oversampling, undersampling)
# - Exploring other evaluation metrics such as F1-score, specificity, or balanced accuracy.
# - Collecting more data if possible to improve model generalization.
