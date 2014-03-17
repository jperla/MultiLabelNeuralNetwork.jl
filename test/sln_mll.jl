using Base.Test

import NeuralNetworks: SLN_MLL, SLN_MLL_Activation,
                       forward_propagate!, calculate_label_probabilities, gradient, zero!,
                       top_features, top_weights, hidden_nodes_table,
                       sigmoid

num_dimensions = 10
num_labels = 3
sln = SLN_MLL(num_dimensions, num_labels, 2)
@assert sln.input_output[1] != 0.0
zero!(sln)
@assert sln.input_output[1] == 0.0

output_labels = ["red", "green", "blue"]
input_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

# All zero weights
x1 = ones(num_dimensions)
output_probabilities = calculate_label_probabilities(sln, x1)
half = (0.5 .* ones(num_labels))
@test output_probabilities == half

# Train the first label up for every input
sln.input_output[:,1] = 1e10
sln.input_output[1,1] = 2e10
# Train the remaining levels down for every input
sln.input_output[:,2] = -1e10
sln.input_output[:,3] = -1e10
output_probabilities = calculate_label_probabilities(sln, x1)
@test output_probabilities[1:end] == [1.0, 0.0, 0.0]
# Top feature for first output label is first input feature
tf1 = top_features(input_names, sln.input_output[:,1])
@test tf1[1] == ("A", 2e10)

zero!(sln)
# Train the first hidden input fully down, second fully up
sln.input_hidden[:,1] = -1
sln.input_hidden[:,2] = 1
# Train the second label based on the second input, the other two based on the first
sln.hidden_output = [1e9 0.0 1e9;
                     0.0 1e9 0.0]
activation = SLN_MLL_Activation(sln)
forward_propagate!(sln, activation, x1)
output_probabilities = calculate_label_probabilities(sln, x1)
@test output_probabilities[1:end] == [0.5, 1.0, 0.5]
@test activation.hidden == [0.0; num_dimensions]
@test sigmoid(activation.output) == output_probabilities

hidden_nodes_table(STDOUT, sln, input_names, output_labels, 6)

