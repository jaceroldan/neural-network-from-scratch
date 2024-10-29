import numpy as np

def hardcoded_np_neuron():
    inputs = [1.0, 2.0, 3.0, 2.5]
    weights = [0.2, 0.8, -0.5, 1.0]
    bias = 2.0

    output = np.dot(weights, inputs) + bias
    return output


def hardcoded_np_layer():
    inputs = [1.0, 2.0, 3.0, 2.5]
    weights = [
        [0.2, 0.8, -0.5, 1],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87],
    ]
    biases = [2.0, 3.0, 0.5]

    layer_outputs = np.dot(weights, inputs) + biases
    return layer_outputs


print('Hardcoded NP Neuron: ', hardcoded_np_neuron())
print('Hardcoded NP Layer: ', hardcoded_np_layer())
