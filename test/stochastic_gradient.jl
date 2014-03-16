using Base.Test

import StochasticGradient: BinaryLogisticRegressionSGD, MultilabelLogisticRegressionSGD,
                           predict, train_samples!, calculate_gradient!
import NeuralNetworks: log_loss

X = [1.0 1 0 0; 1 1 0 0; 0 0 1 1; 0 0 1 1]
Y = [1.0, 1.0, 0.0, 0.0]

@assert size(X, 1) == length(Y)

dimensions = size(X, 2)
weights = randn(dimensions)
gradient = BinaryLogisticRegressionSGD{Float64}(zeros(Float64, dimensions), 1.0)

function dataset_log_loss(X, Y)
    y_hat = Float64[predict(gradient, weights, X[i,:][:]) for i in 1:length(Y)]
    loss = log_loss(Y, y_hat)
    return loss
end

loss = 1.0
for j in 1:100
    loss = dataset_log_loss(X, Y)
    if j % 10 == 1
        @printf("Iteration %i (loss %4f)\n", j, loss)
    end

    for i in 1:length(Y)
        train_samples!(gradient, weights, X[i,:], Y[i], j)
    end
end
@test loss < 0.05



