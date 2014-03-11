include("abstract_neural_networks.jl")

#####################################
# Weights for the neural network
#####################################

type SLN_MLL <: NeuralNetworkStorage
    # single layer neural network for multi label learning with skip level weights
    input_hidden::Weights # weights to calculate hidden layer
    hidden_output::Weights # weights to teh final layer
    input_output::Weights # skip-level weights direction from input to outputs
end

#####################################
# Constructor
#####################################

function SLN_MLL(num_dimensions::Int, num_labels::Int, num_hidden::Int)
    input_output = randn(num_dimensions, num_labels)
    input_hidden = randn(num_dimensions, num_hidden)
    hidden_output = randn(num_hidden, num_labels)
    SLN_MLL(input_hidden, hidden_output, input_output)
end

#####################################
# Size helper functions
#####################################

num_dimensions(sln::SLN_MLL) = size(sln.input_output, 1)
num_hidden(sln::SLN_MLL) = size(sln.input_hidden, 2)
num_output(sln::SLN_MLL) = size(sln.input_output, 2)
num_labels(sln::SLN_MLL) = num_output(sln)

#####################################
# Vectorized Link functions
#####################################

relu(x) = max(0, x) # rectified linear units
@vectorize_1arg Number relu

sigmoid(x) = 1.0 / (1.0 + e^(-x))
@vectorize_1arg Number sigmoid

#####################################
# Classification / Testing
#####################################

function forward_propagate(sln::SLN_MLL, x::Sample)
    @assert length(x) == num_dimensions(sln)
    hidden = relu(x' * sln.input_hidden)
    skip_activation = x' * sln.input_output
    hidden_activation = hidden * sln.hidden_output
    label_probabilities = sigmoid(hidden_activation .+ skip_activation)
    @assert length(label_probabilities) == num_labels(sln)
    return label_probabilities
end

#####################################
# Training
#####################################

function gradient(sln::SLN_MLL, x::Sample)
    # TODO: use backprop
    # this needs to return a gradient for each output label, right?
    @assert num_dimensions(sln) == length(x)
    g = zeros(num_labels(sln), num_dimensions(sln))                                                                   
    @assert size(g) == (num_labels(sln), num_dimensions(sln))                                                         
    return g                                          
end
