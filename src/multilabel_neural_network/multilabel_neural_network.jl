import NeuralNetworks: SLN_MLL, SLN_MLL_Activation, SLN_MLL_Deltas, SLN_MLL_Derivatives,
                       num_labels,
                       calculate_label_probabilities!, back_propagate!,
                       fill!, flat_weights!, flat_weights_length

import StochasticGradient: StochasticGradientDescent,
                           predict!, train_samples!, calculate_gradient!, regularization

##########################################
# Multilabel NN Types
##########################################

immutable type MultilabelSLNSGD{T} <: StochasticGradientDescent{T}
    scratch_gradient::Vector{T}
    num_labels::Int
    initial_learning_rate::T
    rih::T
    rho::T
    rio::T
    dropout::Bool
    sln::SLN_MLL
    activation::SLN_MLL_Activation
    deltas::SLN_MLL_Deltas
    derivatives::SLN_MLL_Derivatives
end

immutable type MultilabelSLNAdaGrad{T} <: StochasticGradientDescent{T}
    scratch_gradient::Vector{T}
    num_labels::Int
    initial_learning_rate::T
    rih::T
    rho::T
    rio::T
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

MultilabelSLNSGD{T}(sln::SLN_MLL{T};
                    initial_learning_rate::T=0.5,
                    rih::T=0,
                    rho::T=0,
                    rio::T=0,
                    dropout::Bool=false) = 
                    MultilabelSLNSGD{T}(zeros(T, flat_weights_length(sln)),
                                        num_labels(sln), initial_learning_rate,
                                        rih, rho, rio,
                                        dropout,
                                        sln, SLN_MLL_Activation(sln), SLN_MLL_Deltas(sln), SLN_MLL_Derivatives(sln))

MultilabelSLNAdaGrad{T}(sln::SLN_MLL{T};
                        initial_learning_rate::T=0.5, 
                        rih::T=0,
                        rho::T=0,
                        rio::T=0,
                        dropout::Bool=false) = 
                        MultilabelSLNAdaGrad{T}(zeros(T, flat_weights_length(sln)),
                                                num_labels(sln), initial_learning_rate,
                                                rih, rho, rio,
                                                dropout,
                                                zeros(T, length(flat_weights_length(sln))),
                                                sln, SLN_MLL_Activation(sln), SLN_MLL_Deltas(sln), SLN_MLL_Derivatives(sln))

##########################################
# Connections to the neural network
##########################################

function regularization{T}(g::MultilabelSLN{T}, weights::AbstractVector{T}, i::Int)
    io = length(g.sln.input_output)
    ih = length(g.sln.input_hidden)
    ho = length(g.sln.hidden_output)

    if i <= io
        regularization_constant = g.rio
    elseif i <= io + ih
        regularization_constant = g.rih
    elseif i <= io + ih + ho
        regularization_constant = g.rho
    else
       throw(DomainError())
    end
    return 2 * regularization_constant * weights[i]
end

function predict!{T}(g::MultilabelSLN{T}, weights::AbstractArray{T}, X::AbstractMatrix{T}, y_hat::AbstractMatrix{T}, i::Int)
    fill!(g.sln, weights)
    calculate_label_probabilities!(g.sln, g.activation, X, y_hat, i)
end

function predict!{T}(g::MultilabelSLN{T}, weights::AbstractArray{T}, X::AbstractMatrix{T}, y_hat::AbstractMatrix{T})
    fill!(g.sln, weights)
    @assert size(X, 1) == size(y_hat, 1)
    for i in 1:size(X, 1)
        calculate_label_probabilities!(g.sln, g.activation, X, sub(y_hat, (i, 1:size(y_hat, 2))), i)
    end
end

function calculate_gradient!{T<:FloatingPoint}(g::MultilabelSLN{T}, weights::AbstractArray{T}, X::AbstractMatrix{T}, Y::AbstractMatrix{T}, i::Int)
    fill!(g.sln, weights)
    back_propagate!(g.sln, g.activation, g.deltas, g.derivatives, X, Y, i, g.dropout)
    flat_weights!(g.derivatives, g.scratch_gradient)
end

