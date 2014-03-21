abstract GradientScratch{T<:FloatingPoint}
# GradientScratch should have the following field:
#   scratch_gradient::Vector{T}
#   initial_learning_rate::T

abstract StochasticGradientDescent{T} <: GradientScratch{T}

abstract AdaGrad{T} <: StochasticGradientDescent{T}
# AdaGrad types expected to also have
#   diagonal_sum_of_gradients::{T}

