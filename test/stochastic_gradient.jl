using Base.Test

import StochasticGradient: BinaryLogisticRegressionSGD, MultilabelLogisticRegressionSGD, StochasticGradientDescent,
                           BinaryLogisticRegressionAdaGrad, MultilabelLogisticRegressionAdaGrad,
                           predict, train_samples!, calculate_gradient!
import NeuralNetworks: log_loss, flat_weights, SLN_MLL
import MultilabelNeuralNetwork: MultilabelSLN, MultilabelSLNSGD, MultilabelSLNAdaGrad

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

function num_labels{T}(g::MultilabelSLN{T})
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
        if (j % int(num_iter / 10) == 1)
            @printf("Epoch %i (loss %4f)\n", j, loss)
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
@printf("Binary LR SGD\n")
bweights = randn(dimensions)
blrsgd = BinaryLogisticRegressionSGD{Float64}(zeros(Float64, dimensions), 1.0)
@time learn(blrsgd, BX, BY, bweights, 100)

# Binary LR AdaGrad
@printf("Binary LR Adagrad\n")
bweights = randn(dimensions)
blrada = BinaryLogisticRegressionAdaGrad{Float64}(zeros(Float64, dimensions), 1.0, ones(Float64, dimensions))
@time learn(blrada, BX, BY, bweights, 20)

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
@printf("Multilabel LR SGD\n")
mweights = randn(dimensions * nlabels)
mlrsgd = MultilabelLogisticRegressionSGD{Float64}(zeros(Float64, length(mweights)), nlabels, 1.0)
@time learn(mlrsgd, MX, MY, mweights, 100)

# Multilabel LR AdaGrad
@printf("Multilabel LR AdaGrad\n")
mweights = randn(dimensions * nlabels)
mlrada = MultilabelLogisticRegressionAdaGrad{Float64}(zeros(Float64, length(mweights)), nlabels, 1.0, ones(Float64, length(mweights)))
@time learn(mlrada, MX, MY, mweights, 100)

##############################################################
#  Neural Network
##############################################################

@printf("SLN MLL SGD\n")
sln = SLN_MLL(dimensions, nlabels, 2)
mweights = flat_weights(sln)
slnmllsgd = MultilabelSLNSGD{Float64}(zeros(Float64, length(mweights)), nlabels, 1.0, sln)
@time learn(slnmllsgd, MX, MY, mweights, 100)

@printf("SLN MLL AdaGrad\n")
sln = SLN_MLL(dimensions, nlabels, 2)
mweights = flat_weights(sln)
slnmllada = MultilabelSLNAdaGrad{Float64}(zeros(Float64, length(mweights)), nlabels, 1.0, sln, ones(Float64, length(mweights)))
@time learn(slnmllada, MX, MY, mweights, 100)

