include("abstract_neural_networks.jl")

function zero!(nn::NeuralNetworkStorage)
    # Zeroes out all of the Weights in the neural network
    fields = names(nn)
    types = typeof(nn).types
    for (t,f) in zip(types, fields)
        if t <: Weights
            fill!(nn.(f), 0)
        end
    end
end

#####################################
# Vectorized Link functions
#####################################

relu(x) = max(0, x) # rectified linear units
@vectorize_1arg Number relu

sigmoid(x) = 1.0 / (1.0 + e^(-x))
@vectorize_1arg Number sigmoid

function sigmoid_prime(x)
    return sigmoid(x) * (1 - sigmoid(x))
end

#####################################
# Inspection, Debugging,  and Reporting
#####################################

function top_features{T <: String}(features::Vector{T}, weights::Vector{Weight})
    # Accepts an array of weights and an array of human-readable names for the features
    # Returns a sorted list of top (absolute value) size weights (only the top N)
    @assert length(weights) == length(features)
    top = [(features[i], w) for (i, w) in top_weights(weights)]
    return top
end

function top_weights(weights::Vector{Weight})
    top = [(i, w) for (_, i, w) in sort(collect(zip(abs(weights), [1:length(weights)], weights)), rev=true)]
    return top
end
