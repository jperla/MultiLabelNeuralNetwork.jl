using Base.Test

import NeuralNetworks: SLN_MLL, forward_propagate, gradient, zero!
import CheckGrad: checkgrad, approximately_one

num_dimensions = 10
num_labels = 3
sln = SLN_MLL(num_dimensions, num_labels, 2)
@assert sln.input_output[1] != 0.0
zero!(sln)
@assert sln.input_output[1] == 0.0

# All zero weights
x1 = ones(num_dimensions)
output_probabilities = forward_propagate(sln, x1)
half = (0.5 .* ones(num_labels))'
@test output_probabilities == half

# Train the first label up for every input
sln.input_output[:,1] = 1e10
# Train the remaining levels down for every input
sln.input_output[:,2] = -1e10
sln.input_output[:,3] = -1e10
output_probabilities = forward_propagate(sln, x1)
@test output_probabilities[1:end] == [1.0, 0.0, 0.0]

zero!(sln)
# Train the first hidden input fully down, second fully up
sln.input_hidden[:,1] = -1e10
sln.input_hidden[:,2] = 1e10
# Train the second label based on the second input, the other two based on the first
sln.hidden_output = [1e9 0.0 1e9;
                     0.0 1e9 0.0]
output_probabilities = forward_propagate(sln, x1)
@test output_probabilities[1:end] == [0.5, 1.0, 0.5]

