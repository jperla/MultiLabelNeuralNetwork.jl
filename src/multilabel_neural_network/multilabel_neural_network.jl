import NeuralNetworks: SLN_MLL, SLN_MLL_Activation, SLN_MLL_Deltas, SLN_MLL_Derivatives,
                       calculate_label_probabilities!, back_propagate!,
                       fill!, flat_weights!

import StochasticGradient: StochasticGradientDescent,
                           predict!, train_samples!, calculate_gradient!

type MultilabelSLNSGD{T} <: StochasticGradientDescent{T}
    scratch_gradient::Vector{T}
    num_labels::Int
    initial_learning_rate::T
    sln::SLN_MLL
    activation::SLN_MLL_Activation
    deltas::SLN_MLL_Deltas
    derivatives::SLN_MLL_Derivatives
    regularization_constant::T
end

type MultilabelSLNAdaGrad{T} <: StochasticGradientDescent{T}
    scratch_gradient::Vector{T}
    num_labels::Int
    initial_learning_rate::T
    sln::SLN_MLL
    diagonal_sum_of_gradients::Vector{T}
    activation::SLN_MLL_Activation
    deltas::SLN_MLL_Deltas
    derivatives::SLN_MLL_Derivatives
    regularization_constant::T
end

typealias MultilabelSLN{T} Union(MultilabelSLNSGD{T}, MultilabelSLNAdaGrad{T})

function predict!{T}(g::MultilabelSLN{T}, weights::Vector{T}, X::AbstractMatrix{T}, y_hat::AbstractMatrix{T}, i::Int)
    fill!(g.sln, weights)
    calculate_label_probabilities!(g.sln, X, y_hat, i, 0)
end

function calculate_gradient!{T<:FloatingPoint}(g::MultilabelSLN{T}, weights::Vector{T}, X::AbstractMatrix{T}, Y::AbstractMatrix{T}, i::Int, dropout::Int)
    fill!(g.sln, weights)
    back_propagate!(g.sln, g.activation, g.deltas, g.derivatives, X, Y, i, dropout::Int64)
    flat_weights!(g.derivatives, g.scratch_gradient)
end

