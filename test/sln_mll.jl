using Base.Test

import NeuralNetworks: SLN_MLL, SLN_MLL_Activation,
                       forward_propagate!, calculate_label_probabilities!, zero!,
                       top_features, top_weights, hidden_nodes_table,
                       sigmoid

TESTT = Float64
num_dimensions = 10
num_labels = 3
sln = SLN_MLL(TESTT, num_dimensions, num_labels, 2)
@assert sln.input_output[1] != 0.0
zero!(sln)
@assert sln.input_output[1] == 0.0

output_labels = ["red", "green", "blue"]
input_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

# All zero weights
x1 = ones((1, num_dimensions))

x1 = sparse(x1)
output_probabilities = zeros(TESTT, 3)
calculate_label_probabilities!(sln, x1, output_probabilities, 1)
half = (0.5 .* ones(num_labels))
@test output_probabilities == half

# Train the first label up for every input
sln.input_output[:,1] = 1e10
sln.input_output[1,1] = 2e10
# Train the remaining levels down for every input
sln.input_output[:,2] = -1e10
sln.input_output[:,3] = -1e10
output_probabilities = zeros(TESTT, 3)
calculate_label_probabilities!(sln, x1, output_probabilities, 1)
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
forward_propagate!(sln, activation, x1, 1)
output_probabilities = zeros(TESTT, 3)
calculate_label_probabilities!(sln, x1, output_probabilities, 1)

# Check that dense calculations are the same as sparse
x1 = full(x1)
forward_propagate!(sln, activation, x1, 1)
dense_probabilities = zeros(TESTT, 3)
calculate_label_probabilities!(sln, x1, dense_probabilities, 1)
@test output_probabilities == dense_probabilities


@test activation.hidden == [0.0; num_dimensions]
@test sigmoid(activation.output) == output_probabilities
@test output_probabilities[1:end] == [0.5, 1.0, 0.5]

hidden_nodes_table(STDOUT, sln, input_names, output_labels, 6)

