abstract GradientScratch{T}
# GradientScratch should have the following field:
#   scratch_gradient::Vector{T}
#   initial_learning_rate::Float64

abstract StochasticGradientDescent{T} <: GradientScratch{T}

abstract AdaGrad{T} <: StochasticGradientDescent{T}
# AdaGrad types expected to also have
#   diagonal_sum_of_gradients::{T}

#############################################
# GradientScratch helpers
#############################################

length_of_weight_updates(g) = length(g.scratch_gradient)

#############################################
# Learning Rates
#############################################

function learning_rate{T}(g::StochasticGradientDescent{T}, i::Int, t::Int)
    return (g.initial_learning_rate / sqrt(t))
end

function learning_rate{T}(g::AdaGrad{T}, i::Int, t::Int)
    return g.initial_learning_rate / sqrt(g.diagonal_sum_of_gradients[i])
end

#############################################
# Save Gradients
#############################################

function save_gradient!{T}(g::AdaGrad{T})
    # Keep track of only diagonal entries as an approximation in AdaGrad
    s = sum(g.scratch_gradient * g.scratch_gradient', 1)
    @assert length(s) == length(g.diagonal_sum_of_gradients)
    for i in 1:length(s)
        g.diagonal_sum_of_gradients[i] += s[i]
    end
    return nothing
end

function save_gradient!(g)
    # Normally a no-op.  This is not a no-op in AdaGrad
    return nothing
end

#############################################
# Update Weights
#############################################

function update_weights!{T}(g::GradientScratch{T}, weights::Vector{T}, t::Int)
    for i in 1:length(weights)
        weights[i] = weights[i] + (learning_rate(g, i, t) .* g.scratch_gradient[i])
    end
end

#############################################
# Training
#############################################

function calculate_gradient_and_update_weights!{T, Y}(g::GradientScratch{T}, weights::Vector{T}, x::Vector{T}, y::Y, t::Int)
    calculate_gradient!(g, weights, x, y) # fill scratch gradient
    save_gradient!(g)
    update_weights!(g, weights, t)
end

function train_samples!{T, Y}(gradient::GradientScratch{T},
			      weights::Vector{T},
                              X::Matrix{T}, y::Y,
                              t::Int # iteration number
                             )
    # Accepts one stochastic sample from the distribution, and updates the weight vector given gradient function.
    # Uses the scratch_gradient as a workspace to avoid memory allocation.
    # We assume that x already has an intercept term (constant 1) as the first value.
    @assert length(weights) == length_of_weight_updates(gradient)
    for i in size(X, 1)
        calculate_gradient_and_update_weights!(gradient, weights, X[i,:][:], y, t)
    end
end

#############################################
# Concrete Logistic Regression SGD
#############################################

## Binary LR
type BinaryLogisticRegressionSGD{T} <: StochasticGradientDescent{T}
    scratch_gradient::Vector{T}
    initial_learning_rate::Float64
end
BinaryLogisticRegressionSGD(dims::Int, eta0::Float64) = BinaryLogisticRegressionSGD{T}(zeros(dims, T), eta0)

## Multilabel LR

type MultilabelLogisticRegressionSGD{T} <: StochasticGradientDescent{T}
    scratch_gradient::Vector{T}
    num_labels::Int
    initial_learning_rate::Float64
end
MultilabelLogisticRegressionSGD(dims::Int, num_labels::Int, eta0::Float64) = MultilabelLogisticRegressionSGD{T}(zeros(dims, T), num_labels, eta0)

## AdaGrad Binary LR

## Binary LR
type BinaryLogisticRegressionAdaGrad{T} <: AdaGrad{T}
    scratch_gradient::Vector{T}
    initial_learning_rate::Float64
    diagonal_sum_of_gradients::Vector{T}
end
BinaryLogisticRegressionAdaGrad(dims::Int, eta0::Float64) = BinaryLogisticRegressionAdagrad{T}(zeros(T, dimsT), eta0, zeros(T, dims))

# Multilabel LR
type MultilabelLogisticRegressionAdaGrad{T} <: AdaGrad{T}
    scratch_gradient::Vector{T}
    num_labels::Int
    initial_learning_rate::Float64
    diagonal_sum_of_gradients::Vector{T}
end
MultilabelLogisticRegressionAdaGrad(dims::Int, num_labels::Int, eta0::Float64) = MultilabelLogisticRegressionSGD{T}(zeros(T,dims), num_labels, eta0, zeros(T, dims))

## Binary and multilabel gradients/predictions

sigmoid(x) = 1.0 / (1.0 + e^(-x))

typealias BinaryLogisticRegression{T} Union(BinaryLogisticRegressionSGD{T}, BinaryLogisticRegressionAdaGrad{T})
typealias MultilabelLogisticRegression{T} Union(MultilabelLogisticRegressionSGD{T}, MultilabelLogisticRegressionAdaGrad{T})

function predict{T}(g::BinaryLogisticRegression{T}, weights::Vector{T}, x::Vector{T})
    return sigmoid(sum(weights .* x))
end

function calculate_gradient!(g::BinaryLogisticRegression{Float64}, weights::Vector{Float64}, x::Vector{Float64}, y::Float64)
    @assert length(weights) == length(x) == length(g.scratch_gradient)
    g.scratch_gradient = (y - predict(g, weights, x)) .* x
end

function predict{T}(g::MultilabelLogisticRegression{T}, weights::Vector{T}, x::Vector{T})
    prediction = zeros(Float64, g.num_labels)
    lx = length(x)
    for i in 1:g.num_labels
        r = (((i - 1) * lx) + 1):(((i - 1) * lx) + lx)
        prediction[i] = sigmoid(sum(weights[r] .* x))
    end
    return prediction
end

function calculate_gradient!(g::MultilabelLogisticRegression{Float64}, 
                             weights::Vector{Float64}, x::Vector{Float64}, y::Vector{Float64})
    @assert length(weights) == (length(x) * g.num_labels) == length(g.scratch_gradient)
    lx = length(x)
    for i in 1:g.num_labels
        r = (((i - 1) * lx) + 1):(((i - 1) * lx) + lx)
        hx = sigmoid(sum(weights[r] .* x))
        g.scratch_gradient[r] = (y[i] - hx) .* x
    end
end

