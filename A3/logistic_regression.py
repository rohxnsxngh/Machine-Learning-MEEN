import numpy as np

def sigmoid(z):
    """
    Sigmoid activation function.
    """
    return 1 / (1 + np.exp(-z))

def fit(X_data, y_data, eta, n_epochs):
    """
    Trains weights of a logistic regression model using gradient descent.
    
    Parameters:
    X_data : numpy.ndarray
        A 2D array of feature values for training.
    y_data : numpy.ndarray
        A 1D array of labels for training.
    eta : float
        Learning rate for gradient descent.
    n_epochs : int
        Number of epochs (iterations) for training.
    
    Returns:
    weights : numpy.ndarray
        Learned weights for the logistic regression model.
    """
    # Add a column of ones for the intercept term
    X_data = np.column_stack((np.ones(X_data.shape[0]), X_data))
    
    # Initialize weights to zeros
    weights = np.zeros(X_data.shape[1])
    
    for epoch in range(n_epochs):
        for i in range(X_data.shape[0]):
            z = np.dot(X_data[i], weights)
            h = sigmoid(z)
            gradient = (y_data[i] - h) * X_data[i]
            weights += eta * gradient
    
    return weights

def predict(x, weights):
    """
    Makes a binary class prediction based on a logistic regression model.

    Parameters:
    x : numpy.ndarray
        1D array of feature values for prediction.
    weights : numpy.ndarray
        Learned weights for the logistic regression model.

    Returns:
    class_label : int
        Predicted class label of 0 or 1.
    """
    # Add a bias term for the intercept
    x = np.insert(x, 0, 1)
    
    z = np.dot(x, weights)
    h = sigmoid(z)
    
    if h >= 0.5:
        return 1
    else:
        return 0
