import NeuralNetworks: SLN_MLL, SLN_MLL_Activation, SLN_MLL_Deltas, SLN_MLL_Derivatives,
                       calculate_label_probabilities, back_propagate!,
                       fill!, flat_weights!

import StochasticGradient: StochasticGradientDescent,
                           predict, train_samples!, calculate_gradient!

type MultilabelSLNSGD{T} <: StochasticGradientDescent{T}
    scratch_gradient::Vector{T}
    num_labels::Int
    initial_learning_rate::Float64
    sln::SLN_MLL
    activation::SLN_MLL_Activation
    deltas::SLN_MLL_Deltas
    derivatives::SLN_MLL_Derivatives
    regularization_constant::Float64
end

type MultilabelSLNAdaGrad{T} <: StochasticGradientDescent{T}
    scratch_gradient::Vector{T}
    num_labels::Int
    initial_learning_rate::Float64
    sln::SLN_MLL
    diagonal_sum_of_gradients::Vector{T}
    activation::SLN_MLL_Activation                                                                                     
    deltas::SLN_MLL_Deltas                                                                                             
    derivatives::SLN_MLL_Derivatives
    regularization_constant::Float64 
end

typealias MultilabelSLN{T} Union(MultilabelSLNSGD{T}, MultilabelSLNAdaGrad{T})

function predict{T}(g::MultilabelSLN{T}, weights::Vector{T}, x::Vector{T})
    fill!(g.sln, weights)
    return calculate_label_probabilities(g.sln, x)
end

function calculate_gradient!(g::MultilabelSLN{Float64}, weights::Vector{Float64}, x::Vector{Float64}, y::Vector{Float64})
    fill!(g.sln, weights)
    back_propagate!(g.sln, g.activation, g.deltas, g.derivatives, x, y)
    flat_weights!(g.derivatives, g.scratch_gradient)
end

