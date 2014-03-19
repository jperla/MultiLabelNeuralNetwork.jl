using Base.Test

import Thresholds: micro_f1_calculate, macro_f1_calculate, accuracy_calculate
import StochasticGradient: BinaryLogisticRegressionSGD, MultilabelLogisticRegressionSGD, StochasticGradientDescent,
                           BinaryLogisticRegressionAdaGrad, MultilabelLogisticRegressionAdaGrad,
                           predict!, train_samples!, calculate_gradient!
import NeuralNetworks: log_loss, flat_weights!, 
                       SLN_MLL, SLN_MLL_Activation, SLN_MLL_Deltas, SLN_MLL_Derivatives
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

function dataset_log_loss{T}(g, w::Vector{T}, X::Matrix{T}, Y::Matrix{T})
    y_hat = zeros(Float64, (size(Y, 1), num_labels(g)))
    for i in 1:size(Y, 1)
        predict!(g, w, X, sub(y_hat, (i, 1:num_labels(g))), i)
    end
    loss = log_loss(Y, y_hat)
    micro_f1 = micro_f1_calculate(y_hat, Y)
    accuracy = accuracy_calculate(y_hat, Y)
    return loss, micro_f1, accuracy
end

function dataset_log_loss{T}(g, w::Vector{T}, X::Matrix{T}, Y::Vector{T})
    y_hat = zeros(Float64, size(Y, 1))
    for i in 1:size(Y, 1)
        predict!(g, w, X, y_hat, i)
    end
    loss = log_loss(Y, y_hat)
    return loss, 0.0, 0.0
end

function learn(g, X, Y, w, num_iter=100)
    loss = 1.0
    for j in 1:num_iter
        
        loss, micro_f1, accuracy= dataset_log_loss(g, w, X, Y)
        if (num_iter < 10 || (j % int(num_iter / 10) == 1))
            @printf("Epoch %i (loss %4f)", j, loss)
            @printf("\t Micro_F1: %4f,  Accuracy: %4f \n", micro_f1, accuracy )
            #@printf("\t Weights: %s \n", w')
        end

        for i in 1:size(Y, 1)
            train_samples!(g, w, X, Y, i:i, j)
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
mlrsgd = MultilabelLogisticRegressionSGD{Float64}(zeros(Float64, length(mweights)), nlabels, 1.0, zeros(Float64, nlabels))
@time learn(mlrsgd, MX, MY, mweights, 100)

# Multilabel LR AdaGrad
@printf("Multilabel LR AdaGrad\n")
mweights = randn(dimensions * nlabels)
mlrada = MultilabelLogisticRegressionAdaGrad{Float64}(zeros(Float64, length(mweights)), nlabels, 1.0, ones(Float64, length(mweights)), zeros(Float64, nlabels))
@time learn(mlrada, MX, MY, mweights, 100)

##############################################################
#  Neural Network
##############################################################

num_hidden = 2

@printf("SLN MLL SGD\n")
sln = SLN_MLL(dimensions, nlabels, num_hidden)
mweights = zeros(Float64, length(sln.input_hidden) + length(sln.input_output) + length(sln.hidden_output))
flat_weights!(sln, mweights)
activation = SLN_MLL_Activation(sln)
deltas = SLN_MLL_Deltas(sln)
derivatives = SLN_MLL_Derivatives(sln)
slnmllsgd = MultilabelSLNSGD{Float64}(zeros(Float64, length(mweights)), nlabels, 1.0, sln, activation, deltas, derivatives, 0.0001)
@time learn(slnmllsgd, MX, MY, mweights, 100)

@printf("SLN MLL AdaGrad\n")
sln = SLN_MLL(dimensions, nlabels, num_hidden)
activation = SLN_MLL_Activation(sln)
deltas = SLN_MLL_Deltas(sln)
derivatives = SLN_MLL_Derivatives(sln)
flat_weights!(sln, mweights)
slnmllada = MultilabelSLNAdaGrad{Float64}(zeros(Float64, length(mweights)), nlabels, 1.0, sln, ones(Float64, length(mweights)), activation, deltas, derivatives, 0.0001)
@time learn(slnmllada, MX, MY, mweights, 100)

