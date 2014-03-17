using Base.Test

import StochasticGradient: BinaryLogisticRegressionSGD, MultilabelLogisticRegressionSGD, StochasticGradientDescent,
                           BinaryLogisticRegressionAdaGrad, MultilabelLogisticRegressionAdaGrad,
                           predict, train_samples!, calculate_gradient!
import NeuralNetworks: log_loss

function num_labels{T}(g::MultilabelLogisticRegressionSGD{T})
    return g.num_labels
end

function num_labels{T}(g::BinaryLogisticRegressionSGD{T})
    return 1
end

function num_labels{T}(g::BinaryLogisticRegressionAdaGrad{T})
    return 1
end

function num_labels{T}(g::MultilabelLogisticRegressionAdaGrad{T})
    return g.num_labels
end

function dataset_log_loss(g, w, X, Y)
    y_hat = zeros(Float64, (size(Y, 1), num_labels(g)))
    for i in 1:size(Y, 1)
        y_hat[i,:] = predict(g, w, X[i,:][:])
    end
    loss = log_loss(Y, y_hat)
    return loss
end

function learn(g, X, Y, w, num_iter=100)
    loss = 1.0
    for j in 1:num_iter
        loss = dataset_log_loss(g, w, X, Y)
        if j % int(num_iter / 10) == 1
            @printf("Iteration %i (loss %4f)\n", j, loss)
        end

        for i in 1:size(Y, 1)
            y = Y[i,:][:]
            if length(y) == 1
                train_samples!(g, w, X[i,:], y[1], j)
            else
                train_samples!(g, w, X[i,:], y, j)
            end
        end
    end
    @test loss < 0.05
end


##############################################################
# Test Binary LR SGD
##############################################################

# Binary Data
BX = [1.0 1 0 0; 1 1 0 0; 0 0 1 1; 0 0 1 1]
BY = [1.0, 1.0, 0.0, 0.0]
@assert size(BX, 1) == length(BY)
dimensions = size(BX, 2)

# Binary LR SGD
bweights = randn(dimensions)
blrsgd = BinaryLogisticRegressionSGD{Float64}(zeros(Float64, dimensions), 1.0)
learn(blrsgd, BX, BY, bweights, 100)

# Binary LR AdaGrad
bweights = randn(dimensions)
blrada = BinaryLogisticRegressionAdaGrad{Float64}(zeros(Float64, dimensions), 1.0, ones(Float64, dimensions))
learn(blrada, BX, BY, bweights, 20)

##############################################################
#  Multilabel
##############################################################

# Multilabel Data
MX = [1.0 1 0 0; 1 1 0 0; 0 0 1 1; 0 0 1 1; 1 1 1 1; 1 1 1 1]
MY = [1.0 0.0  ; 1.0 0.0; 0.0 1.0; 0.0 1.0; 1.0 1.0; 1.0 1.0]
@assert size(MX, 1) == size(MY, 1)
dimensions = size(MX, 2)
nlabels = size(MY, 2)

# Multilabel LR SGD
mweights = randn(dimensions * nlabels)
mlrsgd = MultilabelLogisticRegressionSGD{Float64}(zeros(Float64, length(mweights)), nlabels, 1.0)
learn(mlrsgd, MX, MY, mweights, 100)

# Multilabel LR AdaGrad
mweights = randn(dimensions * nlabels)
mlrada = MultilabelLogisticRegressionAdaGrad{Float64}(zeros(Float64, length(mweights)), nlabels, 1.0, ones(Float64, length(mweights)))
learn(mlrada, MX, MY, mweights, 100)

##############################################################
#  Neural Network
##############################################################

import NeuralNetworks: SLN_MLL,
                       calculate_label_probabilities, back_propagate!,
                       fill!, flat_weights

type MultilabelSLNMLLSGD{T} <: StochasticGradientDescent{T}
    scratch_gradient::Vector{T}
    num_labels::Int
    initial_learning_rate::Float64
    sln::SLN_MLL
end

function predict{T}(g::MultilabelSLNMLLSGD{T}, weights::Vector{T}, x::Vector{T})
    fill!(g.sln, weights)
    return calculate_label_probabilities(g.sln, x)
end

function calculate_gradient!(g::MultilabelSLNMLLSGD{Float64}, weights::Vector{Float64}, x::Vector{Float64}, y::Vector{Float64})
    fill!(g.sln, weights)
    derivatives = back_propagate!(g.sln, x, y)
    gradient = flat_weights(derivatives)
    @printf("gradient: %s:", gradient')
    g.scratch_gradient = gradient
end

function num_labels{T}(g::MultilabelSLNMLLSGD{T})
    return g.num_labels
end

sln = SLN_MLL(dimensions, nlabels, 2)
mweights = flat_weights(sln)
slnmllsgd = MultilabelSLNMLLSGD{Float64}(zeros(Float64, length(mweights)), nlabels, 1.0, sln)
learn(slnmllsgd, MX, MY, mweights, 100)


