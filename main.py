# -*- coding: utf-8 -*-
"""main.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Y0QJXplfGS233cwmvc0VubezWBIDygZK
"""

from data_preparation import load_data, prepare_data
from model import neural_network
from utils import predict, calculate_accuracy

if __name__ == "__main__":
    # Load and prepare data
    filepath = "data/bank-full.csv"
    data = load_data(filepath)
    X, Y = prepare_data(data)

    # Train the neural network
    model = neural_network(X, Y, n_h=10, num_iterations=10)

    # Make predictions and calculate accuracy
    predictions = predict(model, X)
    accuracy = calculate_accuracy(predictions, Y)

    print(f"Accuracy: {accuracy}%")