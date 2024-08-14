
Title: Implementation and Evaluation of a Neural Network for Predicting Bank Term Deposit Subscriptions
Abstract
This paper describes the development and evaluation of a neural network model designed to predict customer subscription to a term deposit using a dataset from a bank marketing campaign. The model leverages fundamental neural network principles, including forward and backward propagation, activation functions, and parameter optimization. We analyze the performance of the model in terms of accuracy and discuss its potential applications.

1. Introduction
Predictive modeling plays a crucial role in various business applications, including marketing. This study utilizes a neural network to predict whether a client will subscribe to a term deposit based on demographic and campaign data. The model is built from scratch, providing insights into neural network training and performance evaluation.

2. Dataset Preparation
The dataset used in this study is sourced from the Bank Marketing dataset, which includes attributes related to customer demographics and marketing campaign interactions.

python
Copy code
import pandas as pd
import numpy as np

# Load dataset
a = pd.read_csv("../input/bank-marketing/bank-full.csv")
To streamline the dataset, several columns are dropped due to their irrelevance or redundancy:

python
Copy code
train = a.drop(['job', 'marital', 'education', 'contact', 'month', 'poutcome', 'default', 'balance', 'housing', 'loan', 'contact', 'month', 'duration', 'pdays', 'previous', 'poutcome'], axis=1)
The target variable y is binary-encoded for classification purposes:

python
Copy code
train['y'].replace({'no': 0, 'yes': 1}, inplace=True)
The dataset is then split into features (X) and the target variable (Y):

python
Copy code
X = train.drop(['y'], axis=1)
Y = train['y']
3. Neural Network Architecture
The neural network is structured with one input layer, two hidden layers, and one output layer:

Input Layer: Features from the dataset
Hidden Layers: 10 nodes each
Output Layer: Single node for binary classification
Network Parameters Initialization:

python
Copy code
def define_network_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_h, n_h) * 0.01
    b2 = np.zeros((n_h, 1))
    W3 = np.random.randn(n_y, n_h) * 0.01
    b3 = np.zeros((n_y, 1))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
4. Forward Propagation
Forward propagation involves calculating the activations for each layer using the sigmoid activation function:

python
Copy code
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def forward_propagation(X, params):
    Z1 = np.dot(params['W1'], X) + params['b1']
    A1 = sigmoid(Z1)

    Z2 = np.dot(params['W2'], A1) + params['b2']
    A2 = sigmoid(Z2)

    Z3 = np.dot(params['W3'], A2) + params['b3']
    A3 = sigmoid(Z3)
    return {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3, "A3": A3}
5. Error Computation
The error is computed using cross-entropy loss:

python
Copy code
def compute_error(Predicted, Actual):
    logprobs = np.multiply(np.log(Predicted), Actual) + np.multiply(np.log(1 - Predicted), 1 - Actual)
    cost = -np.sum(logprobs) / Actual.shape[1]
    return np.squeeze(cost)
6. Backward Propagation
Backward propagation calculates the gradients of the loss function with respect to the model parameters:

python
Copy code
def backward_propagation(params, activations, X, Y):
    m = X.shape[1]

    dZ3 = activations['A3'] - Y
    dW3 = np.dot(dZ3, activations['A2'].T) / m
    db3 = np.sum(dZ3, axis=1, keepdims=True) / m

    dZ2 = np.dot(params['W3'].T, dZ3) * (1 - np.power(activations['A2'], 2))
    dW2 = np.dot(dZ2, activations['A1'].T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    dZ1 = np.dot(params['W2'].T, dZ2) * (1 - np.power(activations['A1'], 2))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3, "db3": db3}
7. Parameter Update
The parameters are updated using gradient descent:

python
Copy code
def update_parameters(params, derivatives, alpha = 1.2):
    params['W1'] = params['W1'] - alpha * derivatives['dW1']
    params['b1'] = params['b1'] - alpha * derivatives['db1']
    params['W2'] = params['W2'] - alpha * derivatives['dW2']
    params['b2'] = params['b2'] - alpha * derivatives['db2']
    params['W3'] = params['W3'] - alpha * derivatives['dW3']
    params['b3'] = params['b3'] - alpha * derivatives['db3']
    return params
8. Training the Neural Network
The neural network is trained over a specified number of iterations:

python
Copy code
def neural_network(X, Y, n_h, num_iterations=100):
    n_x = network_architecture(X, Y)[0]
    n_y = network_architecture(X, Y)[2]

    params = define_network_parameters(n_x, n_h, n_y)
    for i in range(num_iterations):
        results = forward_propagation(X, params)
        error = compute_error(results['A3'], Y)
        derivatives = backward_propagation(params, results, X, Y)
        params = update_parameters(params, derivatives)
    return params
9. Prediction and Accuracy
The model's performance is evaluated using the accuracy metric:

python
Copy code
def predict(parameters, X):
    results = forward_propagation(X, parameters)
    predictions = np.around(results['A3'])
    return predictions

predictions = predict(model, x)
accuracy = float((np.dot(y, predictions.T) + np.dot(1 - y, 1 - predictions.T)) / float(y.size) * 100)
print('Accuracy: %d' % accuracy + '%')
Accuracy: 88%

10. Conclusion
The implemented neural network model achieves an accuracy of 88% on the Bank Marketing dataset, demonstrating its effectiveness in predicting client subscription to term deposits. Future work may include tuning hyperparameters, exploring alternative architectures, and validating the model on different datasets.

References
Bank Marketing Dataset
Ian Goodfellow, Yoshua Bengio, and Aaron Courville. "Deep Learning." MIT Press, 2016.
