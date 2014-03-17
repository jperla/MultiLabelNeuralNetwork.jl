

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

function regularization_constant(g)
   return 0.0
end

function regularization{T}(g::GradientScratch{T}, weights::Vector{T}, i::Int)
    return -2 * regularization_constant(g) * weights[i]
end

function update_weights!{T}(g::GradientScratch{T}, weights::Vector{T}, t::Int)
    for i in 1:length(weights)
        weights[i] = weights[i] + (learning_rate(g, i, t) .* g.scratch_gradient[i]) + regularization(g, weights, i)
    end
end

#############################################
# Training
#############################################

function calculate_gradient_and_update_weights!{T, U}(g::GradientScratch{T}, weights::Vector{T}, X::Matrix{T}, Y::U, i::Int, t::Int)
    @assert i <= size(X, 1)
    calculate_gradient!(g, weights, X, Y, i) # fill scratch gradient
    save_gradient!(g)
    update_weights!(g, weights, t)
end

function train_samples!{T, U}(gradient::GradientScratch{T},
			      weights::Vector{T},
                              X::Matrix{T}, Y::U,
                              r::Range1,
                              t::Int # iteration number
                             )
    # Accepts one stochastic sample from the distribution, and updates the weight vector given gradient function.
    # Uses the scratch_gradient as a workspace to avoid memory allocation.
    # We assume that x already has an intercept term (constant 1) as the first value.
    @assert length(weights) == length_of_weight_updates(gradient)
    for i in r
        calculate_gradient_and_update_weights!(gradient, weights, X, Y, i, t)
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

function predict!{T}(g::BinaryLogisticRegression{T}, weights::Vector{T}, X::Matrix{T}, y::Vector{T}, i::Int)
    y[i] = 0
    for j in 1:length(weights)
        y[i] += weights[j] * X[i,j]
    end
    y[i] = sigmoid(y[i])
end

function calculate_gradient!(g::BinaryLogisticRegression{Float64}, weights::Vector{Float64}, X::Matrix{Float64}, Y::Vector{Float64}, i::Int)
    @assert length(weights) == size(X, 2) == length(g.scratch_gradient)

    # Use Y as scratch space to store prediction
    true_yi = Y[i]
    predict!(g, weights, X, Y, i)
    predicted_yi = Y[i]

    for j in 1:length(weights)
        g.scratch_gradient[j] = (true_yi - predicted_yi) * X[i,j]
    end
    
    # Ensure true labels Y do not change
    Y[i] = true_yi
end

function predict!{T}(g::MultilabelLogisticRegression{T}, weights::Vector{T}, X::Matrix{T}, y::Vector{T}, i::Int)
    @assert length(weights) == (size(X, 2)  * length(y))
    nf = size(X, 2)
    for k in 1:g.num_labels
        y[k] = 0.0
        for j in 1:nf
            nj = ((k - 1) * nf) + j
            y[k] += weights[nj] * X[i, j]
        end
        y[k] = sigmoid(y[k])
    end
end

function calculate_gradient!(g::MultilabelLogisticRegression{Float64}, 
                             weights::Vector{Float64}, X::Matrix{Float64}, Y::Matrix{Float64}, i::Int)
    @assert g.num_labels == size(Y, 2)
    @assert length(weights) == (size(X, 2) * g.num_labels) == length(g.scratch_gradient)
    nf = size(X, 2)
    predicted = zeros(Float64, size(Y, 2))
    predict!(g, weights, X, predicted, i)
    fill!(g.scratch_gradient, -10.9)
    for k in 1:g.num_labels
        for j in 1:nf
            nj = ((k - 1) * nf) + j
            g.scratch_gradient[nj] = (Y[i, k] - predicted[k]) * X[i, j]
        end
    end

    # Always calculate a new gradient
    @assert sum(g.scratch_gradient .== -10.9) == 0
end

