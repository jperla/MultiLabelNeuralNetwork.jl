module StochasticGradient

include("stochastic_gradient/abstract_stochastic_gradient.jl")

importall Base

export BinaryLogisticRegressionSGD, MultilabelLogisticRegressionSGD, StochasticGradientDescent,
       BinaryLogisticRegressionAdaGrad, MultilabelLogisticRegressionAdaGrad,
       predict, train_samples!, calculate_gradient!,
       read_weights, calculate_losses, regularization

include("stochastic_gradient/stochastic_gradient.jl")
include("stochastic_gradient/analysis.jl")

end # module StochasticGradient
