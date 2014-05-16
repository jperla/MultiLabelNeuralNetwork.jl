using Base.Test

import NeuralNetworks: SLN_MLL, SLN_MLL_Activation, SLN_MLL_Deltas, SLN_MLL_Derivatives,
                       old_back_propagate!, back_propagate!, 
                       calculate_label_probabilities!, zero!,
                       top_features, top_weights, hidden_nodes_table,
                       sigmoid, TanhLinkFunction, SigmoidLinkFunction, RectifiedLinearUnitLinkFunction

TESTT = Float64
num_dimensions = 10
num_labels = 3
num_hidden = 2
sln = SLN_MLL(TESTT, num_dimensions, num_labels, num_hidden, RectifiedLinearUnitLinkFunction(), SigmoidLinkFunction())
@assert sln.input_output[1] != 0.0
zero!(sln)
@assert sln.input_output[1] == 0.0

output_labels = ["red", "green", "blue"]
input_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

# All zero weights
x_dense = ones((1, num_dimensions))
x_sparse = sparse(x_dense)

for x1 in AbstractMatrix[x_dense, x_sparse]
    zero!(sln)
    activation = SLN_MLL_Activation(sln)

    output_probabilities = zeros(TESTT, 3)
    calculate_label_probabilities!(sln, activation, x1, output_probabilities, 1)
    half = (0.5 .* ones(num_labels))
    @printf("h: %s o: %s", half', output_probabilities')
    @test output_probabilities == half

    # Train the first label up for every input
    sln.input_output[:,1] = 1e10
    sln.input_output[1,1] = 2e10
    # Train the remaining levels down for every input
    sln.input_output[:,2] = -1e10
    sln.input_output[:,3] = -1e10
    output_probabilities = zeros(TESTT, 3)
    calculate_label_probabilities!(sln, activation, x1, output_probabilities, 1)
    @test output_probabilities[1:end] == [1.0, 0.0, 0.0]
    # Top feature for first output label is first input feature
    tf1 = top_features(input_names, sln.input_output[:,1])
    @test tf1[1] == ("A", 2e10)

    zero!(sln)
    # Train the first hidden input fully down, second fully up
    sln.input_hidden[:,1] = -1
    sln.input_hidden[:,2] = 1
    # Train the second label based on the second input, the other two based on the first
    sln.hidden_output[:] = [1e9 0.0 1e9;
                            0.0 1e9 0.0]
    output_probabilities = zeros(TESTT, 3)
    calculate_label_probabilities!(sln, activation, x1, output_probabilities, 1)

    # Check that dense calculations are the same as sparse
    x1 = full(x1)
    dense_probabilities = zeros(TESTT, 3)
    calculate_label_probabilities!(sln, activation, x1, dense_probabilities, 1)
    @printf("dp: %s op: %s", dense_probabilities', output_probabilities')
    @test output_probabilities == dense_probabilities

    @printf("ah: %s op: %s", activation.hidden_linked', output_probabilities')
    @printf("ah: %s op: %s", activation.output_linked', output_probabilities')

    """
    # tanh link function
    @test activation.hidden == [-1.71689444187673; 1.71689444187673]
    @test sigmoid(activation.output) == output_probabilities
    @test output_probabilities[1:end] == [0.0, 1.0, 0.0]
    """

    # relu link function
    @test activation.hidden_linked == [0.0; num_dimensions]
    @test activation.output_linked == output_probabilities
    @test output_probabilities[1:end] == [0.5, 1.0, 0.5]

    hidden_nodes_table(STDOUT, sln, input_names, output_labels, 6)
end

#####################################################
# Random matrices should generate identical values 
#  for dense and sparse versions (both input and NN)
#####################################################
random_dense = rand((1, num_dimensions))
random_sparse = sparse(random_dense)
random_labels = rand((1, num_labels))

random_sln = SLN_MLL(TESTT, num_dimensions, num_labels, num_hidden, RectifiedLinearUnitLinkFunction(), SigmoidLinkFunction())
activation = SLN_MLL_Activation(random_sln)

# test forward propagate
dense_output = zeros(TESTT, 3)
calculate_label_probabilities!(random_sln, activation, random_dense, dense_output, 1)

sparse_output = zeros(TESTT, 3)
calculate_label_probabilities!(random_sln, activation, random_sparse, sparse_output, 1)

@printf("s: %s d: %s", sparse_output', dense_output')
@test sparse_output == dense_output

# test back propagate
deltas = SLN_MLL_Deltas(sln)
derivatives = SLN_MLL_Derivatives(sln)
back_propagate!(sln, activation, deltas, derivatives, 
                random_dense, random_labels, 1, false)

old_deltas = SLN_MLL_Deltas(sln)
old_derivatives = SLN_MLL_Derivatives(sln)
old_back_propagate!(sln, activation, old_deltas, old_derivatives, 
                    random_dense, random_labels, 1, false)

@printf("o: %s n: %s", old_derivatives.input_hidden', derivatives.input_hidden')
@test old_derivatives.input_output == derivatives.input_output
@test old_derivatives.input_hidden == derivatives.input_hidden
@test old_derivatives.hidden_output == derivatives.hidden_output

#####################################################
# input->output backprop tests
#####################################################

activation = SLN_MLL_Activation(sln)
xb = ones((1, num_dimensions)) * 0.5
yb = ones((1, num_labels))

for i in 1:num_dimensions, j in 1:num_labels
    zero!(sln)
    sln.input_output[i, j] = 0.5

    deltas = SLN_MLL_Deltas(sln)
    derivatives = SLN_MLL_Derivatives(sln)
    back_propagate!(sln, activation, deltas, derivatives, xb, yb, 1, false)

    predicted_output = zeros(num_labels)
    predicted_output[j] = 0.25
    @test activation.output == predicted_output

    @test derivatives.input_output[i, j] == -0.21891174955710097
end

