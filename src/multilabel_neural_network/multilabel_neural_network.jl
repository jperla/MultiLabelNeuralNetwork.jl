import NeuralNetworks: SLN_MLL,
                       calculate_label_probabilities, back_propagate!,
                       fill!, flat_weights

import StochasticGradient: StochasticGradientDescent,
                           predict, train_samples!, calculate_gradient!

type MultilabelSLNSGD{T} <: StochasticGradientDescent{T}
    scratch_gradient::Vector{T}
    num_labels::Int
    initial_learning_rate::Float64
    sln::SLN_MLL
end

type MultilabelSLNAdaGrad{T} <: StochasticGradientDescent{T}
    scratch_gradient::Vector{T}
    num_labels::Int
    initial_learning_rate::Float64
    sln::SLN_MLL
    diagonal_sum_of_gradients::Vector{T}
end

typealias MultilabelSLN{T} Union(MultilabelSLNSGD{T}, MultilabelSLNAdaGrad{T})

function predict{T}(g::MultilabelSLN{T}, weights::Vector{T}, x::Vector{T})
    fill!(g.sln, weights)
    return calculate_label_probabilities(g.sln, x)
end

function calculate_gradient!(g::MultilabelSLN{Float64}, weights::Vector{Float64}, x::Vector{Float64}, y::Vector{Float64})
    fill!(g.sln, weights)
    derivatives = back_propagate!(g.sln, x, y)
    gradient = flat_weights(derivatives) 
    g.scratch_gradient = gradient
end

