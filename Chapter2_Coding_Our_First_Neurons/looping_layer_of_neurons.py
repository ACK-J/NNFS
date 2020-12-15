inputs = [1,2,3,2.5]

weights = [ [0.2,0.8,-0.5,1.0],
            [0.5,-.91,.26,-.5],
            [-0.26,-0.27,.17,.87] ]

biases = [2,3,0.5]

# Output of current layer
layer_output = []

# For each Neuron
for neuron_weights, neuron_bias in zip(weights, biases):
    # Zeroed output of given neuron_bias
    neuron_output = 0
    # For each input and weight to the neuron
    for n_input, weight in zip(inputs, neuron_weights):
        # multiply this input by associated weight
        # and add to the neuron's output variable
        neuron_output += n_input*weight
    # Add bias
    neuron_output += neuron_bias
    # put neuron's result to the layer's output list
    layer_output.append(neuron_output)

print(layer_output)
