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

function assert_not_NaN(nn::NeuralNetworkStorage)
    # Asserts none of the weights are NaN
    fields = names(nn)
    types = typeof(nn).types
    for (t,f) in zip(types, fields)
        if t <: Array{Float64}
            if !assert_not_NaN(nn.(f))
                return false
            end
        end
    end
    return true
end

function assert_not_NaN(x::Array{Float64})
    for i in 1:length(x)
	if !assert_not_NaN(x[i])
	    return false
	end
    end
    return true
end

function assert_not_NaN(x::Float64)
    if false || isequal(x, NaN)
    	return false
    else
    	return true
    end
end

#####################################
# Vectorized Link functions
#####################################

relu(x) = max(0, x) # rectified linear units
@vectorize_1arg Number relu

base_sigmoid(x) = (1.0 / (1.0 + e^(-x)))

sigmoid(x) = base_sigmoid(x) #((2.0 * base_sigmoid(x)) - 0.5)

@vectorize_1arg Number sigmoid

function sigmoid_prime(x)
    return 1.0 * base_sigmoid(x) * (1.0 - base_sigmoid(x))
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
