function zero!{T}(nn::NeuralNetworkStorage{T})
    # Zeroes out all of the weights in the neural network
    fields = names(nn)
    types = typeof(nn).types
    for (t,f) in zip(types, fields)
        if t <: Array{T}
            fill!(nn.(f), 0)
        end
    end
end

function assert_not_NaN{T}(nn::NeuralNetworkStorage{T})
    # Asserts none of the weights are NaN
    fields = names(nn)
    for f in fields
        if !assert_not_NaN(nn.(f))
            return false
        end
    end
    return true
end

function assert_not_NaN{T}(x::T)
   return true
end

function assert_not_NaN{T<:FloatingPoint}(x::Array{T})
    for i in 1:length(x)
	if !assert_not_NaN(x[i])
	    return false
	end
    end
    return true
end

function assert_not_NaN{T<:FloatingPoint}(x::T)
    if isequal(x, NaN)
    	return false
    else
    	return true
    end
end

#####################################
# Link Functions
#####################################

relu(x) = max(0, x) # rectified linear units

# See http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
lecunn_tanh(x) = 1.7159 * tanh(2x/3) + 10e-5 * x

base_sigmoid{T<:FloatingPoint}(x::T) = (1.0 / (1.0 + e^(-x)))

sigmoid{T<:FloatingPoint}(x::T) = base_sigmoid(x) #((2.0 * base_sigmoid(x)) - 0.5)

@vectorize_1arg Number lecunn_tanh
@vectorize_1arg Number relu
@vectorize_1arg Number sigmoid

function sigmoid_prime{T<:FloatingPoint}(x::T)
    return 1.0 * base_sigmoid(x) * (1.0 - base_sigmoid(x))
end

function lecunn_tanh_prime{T<:FloatingPoint}(x::T)
    return 1.7159 * ( 1 - tanh(2x/3)^2) * 2 + 10e-5
end

function rectified_prime{T<:FloatingPoint}(x::T)
    return 1
end

type TanhLinkFunction <: LinkFunction
end

type SigmoidLinkFunction <: LinkFunction
end

type RectifiedLinearUnitLinkFunction <: LinkFunction
end

function link_function{T}(f::TanhLinkFunction, input::T)
    return lecunn_tanh(input)
end

function link_function{T}(f::SigmoidLinkFunction, input::T)
    return sigmoid(input)
end

function link_function{T}(f::RectifiedLinearUnitLinkFunction, input::T)
    return relu(input)
end

function inverse_link_function{T}(f::TanhLinkFunction, input::T)
    @assert input == 0
    return -Inf
end

function inverse_link_function{T}(f::SigmoidLinkFunction, input::T)
    @assert input == 0
    return -Inf
end

function inverse_link_function{T}(f::RectifiedLinearUnitLinkFunction, input::T)
    @assert input >= 0
    return input
end

function link_function_prime{T}(f::SigmoidLinkFunction, input::T)
    return sigmoid_prime(input)
end

function link_function_prime{T}(f::TanhLinkFunction, input::T)
    return lecunn_tanh_prime(input)
end

function link_function_prime{T}(f::RectifiedLinearUnitLinkFunction, input::T)
    return rectified_prime(input)
end


########################################
# Inspection, Debugging, and Reporting
########################################

function top_features{T<:String, W<:FloatingPoint}(features::Vector{T}, weights::WeightVector{W})
    # Accepts an array of weights and an array of human-readable names for the features
    # Returns a sorted list of top (absolute value) size weights (only the top N)
    @assert length(weights) == length(features)
    top = [(features[i], w) for (i, w) in top_weights(weights)]
    return top
end

function top_weights{W<:FloatingPoint}(weights::WeightVector{W})
    top = [(i, w) for (_, i, w) in sort(collect(zip(abs(weights), [1:length(weights)], weights)), rev=true)]
    return top
end

########################################
# Data Pre-processing
########################################

function whiten{T<:Number}(a::Array{T, 2})
    m = mean(a, 1)
    a = broadcast(-, a, m)
    s = std(a, 1)
    # Do not divide by 0 stddev or we will get NaN!
    for i in 1:length(s)
        if s[i] == 0.0
            s[i] = 1.0
        end
    end
    a = broadcast(/, a, s)
    return a, m, s
end

function whiten{T<:Number}(a::Array{T, 2}, m::Matrix{T}, s::Matrix{T})
    a = broadcast(-, a, m)
    a = broadcast(/, a, s)
    return a
end

function prepend_intercept{T<:Number}(m::Array{T, 2})
    return hcat(ones(T, size(m, 1)), m)
end

