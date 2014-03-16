using Base.Test

import StochasticGradient: BinaryLogisticRegressionSGD, MultilabelLogisticRegressionSGD,
                           predict, train_samples!, calculate_gradient!
import NeuralNetworks: log_loss

function num_labels{T}(g::MultilabelLogisticRegressionSGD{T})
    return g.num_labels
end

function num_labels{T}(g::BinaryLogisticRegressionSGD{T})
    return 1
end

function dataset_log_loss(g, X, Y)
    y_hat = zeros(Float64, (size(Y, 1), num_labels(g)))
    for i in 1:size(Y, 1)
        y_hat[i,:] = predict(g, weights, X[i,:][:])
    end
    loss = log_loss(Y, y_hat)
    return loss
end

# Test Binary LR SGD
X = [1.0 1 0 0; 1 1 0 0; 0 0 1 1; 0 0 1 1]
Y = [1.0, 1.0, 0.0, 0.0]

@assert size(X, 1) == length(Y)

dimensions = size(X, 2)
weights = randn(dimensions)
blrsgd = BinaryLogisticRegressionSGD{Float64}(zeros(Float64, dimensions), 1.0)

loss = 1.0
for j in 1:100
    loss = dataset_log_loss(blrsgd, X, Y)
    if j % 10 == 1
        @printf("Iteration %i (loss %4f)\n", j, loss)
    end

    for i in 1:length(Y)
        train_samples!(blrsgd, weights, X[i,:], Y[i], j)
    end
end
@test loss < 0.05




X = [1.0 1 0 0; 1 1 0 0; 0 0 1 1; 0 0 1 1; 1 1 1 1; 1 1 1 1]
Y = [1.0 0.0  ; 1.0 0.0; 0.0 1.0; 0.0 1.0; 1.0 1.0; 1.0 1.0]

@assert size(X, 1) == size(Y, 1)

dimensions = size(X, 2)
nlabels = size(Y, 2)
weights = randn(dimensions * nlabels)
mlrsgd = MultilabelLogisticRegressionSGD{Float64}(zeros(Float64, length(weights)), nlabels, 1.0)

loss = 1.0
for j in 1:100
    loss = dataset_log_loss(mlrsgd, X, Y)
    if j % 10 == 1
        @printf("Iteration %i (loss %4f)\n", j, loss)
    end

    for i in 1:size(Y, 1)
        train_samples!(mlrsgd, weights, X[i,:], Y[i,:][:], j)
    end
end
@test loss < 0.05



