import numpy as np
import nnfs

nnfs.init()

print(np.random.randn(2, 3))

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases        rows   X   columns
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        pass
