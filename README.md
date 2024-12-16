# Multi-Purpose Neural Network

This repository contains a Python implementation of a flexible and modular neural network designed for both classification and regression tasks.

## Features

- **Dynamic Weight Initialisation:** Uses Xavier initialisation for faster convergence.
- **Customisable Architecture:** Define the number of layers and neurons per layer.
- **Flexible Activation Functions:** Supports Sigmoid, ReLU, and Softmax.
- **Performance Metrics:** Calculate accuracy, precision, and RMSE.
- **Dual Task Support:** Optimised for classification and regression problems.

## Getting Started

### Prerequisites

- Python 3.x
- Required library: `numpy`

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/multi-purpose-nn.git
Install the required dependencies:
bash
pip install numpy
Usage
# Initialising the Neural Network

from multi_purpose_NN import MultiPurposeNN

# Create a neural network instance
nn = MultiPurposeNN(
    alpha=0.01,
    activation_func='ReLU',
    error_func='class_Xentropy',
    n_neurons=10,
    n_features=4,
    n_instances=100,
    n_layers=3
)
Training the Model
nn.train(X_train, y_train, epochs=1000)

# Making Predictions
predictions = nn.predict(X_test)

# Evaluating Performance
Accuracy (for classification):
accuracy = nn.calculate_accuracy(y_true, y_pred)
RMSE (for regression):
rmse = nn.calculate_rmse(y_true, y_pred)

# File Structure

multi_purpose_NN.py: Contains the neural network class implementation.

# Roadmap
Planned future improvements:

Add additional activation functions (e.g., Tanh).
Implement advanced optimisation techniques (e.g., Adam).
Add support for dropout regularisation.
