import numpy as np
import pandas as pd
import ast

# Logistic Regression Functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_loss(y, y_hat):
    m = y.size
    return -(1/m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

def logistic_regression(X, y, num_iterations, learning_rate):
    m, n = X.shape
    w = np.zeros((n, 1))
    b = 0

    # Ensure y is a column vector
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    for i in range(num_iterations):
        # Forward propagation
        z = np.dot(X, w) + b
        y_hat = sigmoid(z)

        # Compute loss
        loss = compute_loss(y, y_hat)

        # Backward propagation
        dw = (1/m) * np.dot(X.T, (y_hat - y))
        db = (1/m) * np.sum(y_hat - y)

        # Update parameters
        w -= learning_rate * dw
        b -= learning_rate * db

        if i % 100 == 0:
            print(f"Iteration {i}: Loss {loss}")

    return w, b


def predict(w, b, X):
    z = np.dot(X, w) + b
    y_hat = sigmoid(z)
    return np.where(y_hat > 0.5, 1, 0)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Example Synthetic Data
np.random.seed(0)
X = np.random.randn(100, 3) # 100 samples, 3 features
y = np.random.randint(0, 2, (100, 1)) # 100 binary labels

# Split the data into training and testing sets
# split_ratio = 0.8
# split = int(split_ratio * X.shape[0])
# X_train, X_test = X[:split], X[split:]
# y_train, y_test = y[:split], y[split:]

df = pd.read_excel('encoding_results.xlsx')
df['Encoding'] = df['Encoding'].apply(ast.literal_eval)
X_train = df['Encoding'].tolist()
y_train = df['Status'].tolist()

X_train = np.array(X_train)
y_train = np.array(y_train)

df = pd.read_excel('encoding_results.xlsx')
df['Encoding'] = df['Encoding'].apply(ast.literal_eval)
X_test = df['Encoding'].tolist()
y_test = df['Status'].tolist()

X_test = np.array(X_test)
y_test = np.array(y_test)

# Train the model
w, b = logistic_regression(X_train, y_train, num_iterations=1000, learning_rate=0.0001)

# Testing the model
y_pred_test = predict(w, b, X_test)
test_accuracy = accuracy(y_test, y_pred_test)

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
