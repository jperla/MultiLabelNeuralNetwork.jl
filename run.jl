#!/usr/bin/env julia
import ArgParse: ArgParseSettings, @add_arg_table, parse_args

import StochasticGradient: train_samples!
import NeuralNetworks: SLN_MLL, SLN_MLL_Activation, SLN_MLL_Deltas, SLN_MLL_Derivatives,
                       read_data, flat_weights, 
                       log_loss, assert_not_NaN
import MultilabelNeuralNetwork: MultilabelSLN, MultilabelSLNAdaGrad, MultilabelSLNSGD, 
                                predict, calculate_gradient!
import Thresholds: accuracy_calculate, micro_f1_calculate


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "dataset"
            help = "dataset we want to use: emotions, scene, yeast"
            arg_type = String
            required = true
        "hidden"
            help = "the number of hidden nodes"
            arg_type = Integer
            required = true
        "--eta0"
            help = "the initial learning rate"
            arg_type = FloatingPoint
            default = 0.3
        "--adagrad"
            help = "the initial learning rate"
            arg_type = Bool
            default = false
        "--epochs", "-e"
            help = "Number of epochs to do"
            arg_type = Integer
            default = 100
    end

    return parse_args(s)
end

function num_labels(g)
    g.num_labels
end

function dataset_log_loss(g, w, X, Y)
    y_hat = zeros(Float64, (size(Y, 1), num_labels(g)))
    for i in 1:size(Y, 1)
        y_hat[i,:] = predict(g, w, X[i,:][:])
    end
    loss = log_loss(Y, y_hat)

    @assert assert_not_NaN(y_hat)
    @assert assert_not_NaN(Y)

    if size(Y, 2) > 1
        micro_f1 = micro_f1_calculate(y_hat, Y)
        accuracy = accuracy_calculate(y_hat, Y)
    else
        micro_f1 = accuracy = 0
    end                                                                                                                                
    return loss, micro_f1, accuracy    
end

function learn(g, w, X, Y; epochs=100, modn=10)
    loss = 1.0
    for e in 1:epochs
        loss, micro_f1, accuracy = dataset_log_loss(g, w, X, Y)
        if ((modn == 1) || (e % modn == 1))
            @printf("Epoch %i (loss %4f): %s", e, loss, w[1:3]')
            @printf("\t Micro_F1: %4f,  Accuracy: %4f \n", micro_f1, accuracy)
        end

        for i in 1:size(Y, 1)
            y = Y[i,:][:]
            train_samples!(g, w, X[i,:], y, e)
        end
    end
end

function whiten{T<:Number}(a::Array{T, 2})
    m = mean(a, 1)
    a = broadcast(-, a, m)
    s = std(a, 1)
    a = broadcast(/, a, s)
    return a, m, s
end

function whiten{T<:Number}(a::Array{T, 2}, m::Matrix{Float64}, s::Matrix{Float64})
    a = broadcast(-, a, m)
    a = broadcast(/, a, s)
    return a
end

function prepend_intercept{T<:Number}(m::Array{T, 2})
    return hcat(ones(T, size(m, 1)), m)
end

parsed_args = parse_commandline()
dataset = parsed_args["dataset"]
nepochs = parsed_args["epochs"]
hidden_nodes = parsed_args["hidden"]
initial_learning_rate = parsed_args["eta0"]
adagrad = parsed_args["adagrad"]

#########################
# Read and cleanup data
#########################
train_features, train_labels = read_data(dataset, "train")
train_features, train_mean, train_std = whiten(train_features)
train_features = prepend_intercept(train_features)

test_features, test_labels = read_data(dataset, "test")
test_features = whiten(test_features, train_mean, train_std)
test_features = prepend_intercept(test_features)

dimensions = size(train_features, 2)
nlabels = size(train_labels, 2)
@assert size(train_labels, 1) == size(train_features, 1)

#########################
# Setup Data and Run
#########################

if adagrad
    @printf("SLN MLL AdaGrad\n")
    sln = SLN_MLL(dimensions, nlabels, hidden_nodes)
    mweights = flat_weights(sln)
    slnmllada = MultilabelSLNAdaGrad{Float64}(zeros(Float64, length(mweights)), nlabels, initial_learning_rate, sln, ones(Float64, length(mweights)), SLN_MLL_Activation(sln), SLN_MLL_Deltas(sln), SLN_MLL_Derivatives(sln))

    @time learn(slnmllada, mweights, train_features, train_labels, epochs=nepochs, modn=1)
else
    @printf("SLN MLL SGD\n")
    sln = SLN_MLL(dimensions, nlabels, hidden_nodes)
    mweights = flat_weights(sln)
    slnmllsgd = MultilabelSLNSGD{Float64}(zeros(Float64, length(mweights)), nlabels, initial_learning_rate, sln, SLN_MLL_Activation(sln), SLN_MLL_Deltas(sln), SLN_MLL_Derivatives(sln))

    @time learn(slnmllsgd, mweights, train_features, train_labels, epochs=nepochs, modn=1)
end

