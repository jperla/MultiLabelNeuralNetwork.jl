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

type SLN_MLL_Activation
    # single layer neural network activation levels after being trained on input
    hidden::Activations
    output::Activations
end

#####################################
# Constructors
#####################################

function SLN_MLL(num_dimensions::Int, num_labels::Int, num_hidden::Int)
    input_output = randn(num_dimensions, num_labels)
    input_hidden = randn(num_dimensions, num_hidden)
    hidden_output = randn(num_hidden, num_labels)
    SLN_MLL(input_hidden, hidden_output, input_output)
end

function SLN_MLL_Activation(sln::SLN_MLL)
    SLN_MLL_Activation(zeros(num_hidden(sln)), zeros(num_output(sln)))
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

function forward_propagate!(sln::SLN_MLL, activation::SLN_MLL_Activation, x::Sample)
    @assert length(x) == num_dimensions(sln)
    activation.hidden = relu(x' * sln.input_hidden)[:]
    skip_input = x' * sln.input_output
    hidden_input = activation.hidden' * sln.hidden_output
    activation.output = sigmoid(hidden_input .+ skip_input)[:]
    @assert length(activation.output) == num_labels(sln)
    return activation
end

function calculate_label_probabilities(sln::SLN_MLL, x::Sample)
    activation = SLN_MLL_Activation(sln)
    forward_propagate!(sln, activation, x)
    return activation.output
end

#####################################
# Training
#####################################

function backpropagate!(sln::SLN_MLL, x::Sample, y::Labels)
    # Modifies the weights in the neural network through backpropagation

    # TODO: calculate

    ################################################################
    #   Calculate delta_k
    #   Calculate delta_j for each interior node
    #   Calculate weight updates
    ################################################################


end



function gradient(sln::SLN_MLL, x::Sample)
    @assert num_dimensions(sln) == length(x)
    g = zeros(num_labels(sln), num_dimensions(sln))

    # TODO: calculate

    # this needs to return a gradient for each output label, right?
    @assert size(g) == (num_labels(sln), num_dimensions(sln))
    return g
end
