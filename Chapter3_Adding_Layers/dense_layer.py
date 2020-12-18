import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases        rows   X   columns
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        #                      row X columns
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases
#  Creating the dataset
X, y = spiral_data(samples=100, classes=3)
# First layer is a dense layer with 2 input features and 3 output featurs
dense1 = Layer_Dense(2,3)
# Make a forward pass
dense1.forward(X)
# Getting the output of the first few samples
print(dense1.output[:5])
