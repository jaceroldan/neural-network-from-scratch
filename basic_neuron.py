
def simple_hard_defined_network():
    inputs = [1, 2, 3]
    weights = [0.2, 0.8, -0.5]
    bias = 2

    output = (inputs[0]*weights[0]
              + inputs[1]*weights[1]
              + inputs[2]*weights[2]
              + bias)
    
    return output


def tri_node_simple_hard_defined_network():
    inputs = [1.0, 2.0, 3.0, 2.5]
    weights1 = [0.2, 0.8, -0.5, 1.0]
    weights2 = [0.5, -0.91, 0.26, -0.5]
    weights3 = [-0.26, -0.27, 0.17, 0.87]

    bias1 = 2.0
    bias2 = 3.0
    bias3 = 0.5

    outputs = [
        inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1,
        inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2,
        inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3,
    ]

    return outputs

# global usage
inputs = [1,2,3,2.5]
weights = [
    [0.2, 0.8, -0.5, 1],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87],
]
biases = [2,3,0.5]


def tri_node_iterative_forward(inputs, weights, biases):
    """
    Note that as this now dynamically performs the matrix
    multiplications needed, it must now account for the correct
    dimensions present in inputs and outputs.

    TODO: Missing checks for list dimensions.
    """
    layer_outputs = []

    # we use "zip" here to extract neuron_weights and biases pairwise,
    # denoting each hidden layer's weights and biases.
    for neuron_weights, neuron_bias in zip(weights, biases):
        neuron_output = 0

        # We use zip here to extract inputs and weights pairwise
        # and sum their products together...
        for n_input, weight in zip(inputs, neuron_weights):
            neuron_output += n_input * weight

        # alongside the bias.
        neuron_output += neuron_bias
        layer_outputs.append(neuron_output)
    
    return layer_outputs


print('Simple pairwise network')
print(simple_hard_defined_network())
print('-'*30)
print()

print('Actual network with 4 inputs and 3 hidden layers (hardcoded)')
print(tri_node_simple_hard_defined_network())
print('-'*30)
print()

print('Dynamic computation of layer outputs, so these now take arguments')
print(tri_node_iterative_forward(inputs, weights, biases))
