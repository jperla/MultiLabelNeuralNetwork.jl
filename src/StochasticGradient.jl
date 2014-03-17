module StochasticGradient

importall Base

export BinaryLogisticRegressionSGD, MultilabelLogisticRegressionSGD,
       BinaryLogisticRegressionAdaGrad, MultilabelLogisticRegressionAdaGrad,
       predict, train_samples!, calculate_gradient!

include("stochastic_gradient/stochastic_gradient.jl")

end # module StochasticGradient
