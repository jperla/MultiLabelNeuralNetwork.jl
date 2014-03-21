import NeuralNetworks: SLN_MLL, SLN_MLL_Activation, SLN_MLL_Deltas, SLN_MLL_Derivatives,
                       num_labels,
                       calculate_label_probabilities!, back_propagate!,
                       fill!, flat_weights!

import StochasticGradient: StochasticGradientDescent,
                           predict!, train_samples!, calculate_gradient!

##########################################
# Multilabel NN Types
##########################################

type MultilabelSLNSGD{T} <: StochasticGradientDescent{T}
    scratch_gradient::Vector{T}
    num_labels::Int
    initial_learning_rate::T
    regularization_constant::T
    dropout::Bool
    sln::SLN_MLL
    activation::SLN_MLL_Activation
    deltas::SLN_MLL_Deltas
    derivatives::SLN_MLL_Derivatives
end

type MultilabelSLNAdaGrad{T} <: StochasticGradientDescent{T}
    scratch_gradient::Vector{T}
    num_labels::Int
    initial_learning_rate::T
    regularization_constant::T
    dropout::Bool
    diagonal_sum_of_gradients::Vector{T}
    sln::SLN_MLL
    activation::SLN_MLL_Activation
    deltas::SLN_MLL_Deltas
    derivatives::SLN_MLL_Derivatives
end

typealias MultilabelSLN{T} Union(MultilabelSLNSGD{T}, MultilabelSLNAdaGrad{T})

##########################################
# Constructors
##########################################

MultilabelSLNSGD{T}(sln::SLN_MLL{T},
                    scratch_gradient::Vector{T};
                    initial_learning_rate::T=0.5,
                    regularization_constant::T=0.0001,
                    dropout::Bool=false) = 
                    MultilabelSLNSGD{T}(scratch_gradient,
                                        num_labels(sln), initial_learning_rate, regularization_constant, dropout,
                                        sln, SLN_MLL_Activation(sln), SLN_MLL_Deltas(sln), SLN_MLL_Derivatives(sln))

MultilabelSLNAdaGrad{T}(sln::SLN_MLL{T},
                        scratch_gradient::Vector{T}; 
                        initial_learning_rate::T=0.5, 
                        regularization_constant::T=0.0001,
                        dropout::Bool=false) = 
                        MultilabelSLNAdaGrad{T}(scratch_gradient, 
                                                num_labels(sln), initial_learning_rate, regularization_constant, dropout,
                                                zeros(T, length(scratch_gradient)),
                                                sln, SLN_MLL_Activation(sln), SLN_MLL_Deltas(sln), SLN_MLL_Derivatives(sln))

##########################################
# Connections to the neural network
##########################################

function regularization{T}(g::MultilabelSLN{T}, weights::Vector{T}, i::Int)
    @printf("Using multilabel regularization! %4f", g.regularization_constant)
    return 2 * g.regularization_constant * weights[i]
end

function predict!{T}(g::MultilabelSLN{T}, weights::Vector{T}, X::AbstractMatrix{T}, y_hat::AbstractMatrix{T}, i::Int)
    fill!(g.sln, weights)
    calculate_label_probabilities!(g.sln, g.activation, X, y_hat, i)
end

function predict!{T}(g::MultilabelSLN{T}, weights::Vector{T}, X::AbstractMatrix{T}, y_hat::AbstractMatrix{T})
    fill!(g.sln, weights)
    @assert size(X, 1) == size(y_hat, 1)
    for i in 1:size(X, 1)
        calculate_label_probabilities!(g.sln, g.activation, X, sub(y_hat, (i, 1:size(y_hat, 2))), i)
    end
end

function calculate_gradient!{T<:FloatingPoint}(g::MultilabelSLN{T}, weights::Vector{T}, X::AbstractMatrix{T}, Y::AbstractMatrix{T}, i::Int)
    fill!(g.sln, weights)
    back_propagate!(g.sln, g.activation, g.deltas, g.derivatives, X, Y, i, g.dropout)
    flat_weights!(g.derivatives, g.scratch_gradient)
end

