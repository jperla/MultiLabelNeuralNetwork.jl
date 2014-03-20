#!/usr/bin/env julia
import ArgParse: ArgParseSettings, @add_arg_table, parse_args

import StochasticGradient: train_samples!, StochasticGradientDescent
import NeuralNetworks: SLN_MLL, SLN_MLL_Activation, SLN_MLL_Deltas, SLN_MLL_Derivatives,
                       read_data, flat_weights!,
                       log_loss, assert_not_NaN
import MultilabelNeuralNetwork: MultilabelSLN, MultilabelSLNAdaGrad, MultilabelSLNSGD,
                                predict!, calculate_gradient!
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
            default = 0.01
        "--adagrad"
            help = "the initial learning rate"
            action = :store_true
        "--time"
            help = "measure timings"
            action = :store_true
        "--epochs", "-e"
            help = "Number of epochs to do"
            arg_type = Integer
            default = 100
	"--regularization", "-r"
	    help = "Regularization constant"
	    arg_type = FloatingPoint
            default = .01
	"--interval", "-i"
	    help = "How frequently to print progress"
	    arg_type = Integer
	    default = 1000
        "--file_prefix", "-f"
	    help = "Save weights at each interval to a file instead of calculating losses and printing to screen"
	    arg_type = String
	    default = ""
    	"--dropout"
            help = "use dropout during training"
            action = :store_true

   end

    return parse_args(s)
end

function num_labels(g)
    g.num_labels
end

function dataset_log_loss{T,U<:FloatingPoint,W<:FloatingPoint}(g, w::Vector{T},
                                                               X::AbstractMatrix{U}, Y::AbstractMatrix{W},
                                                               y_hat::AbstractMatrix{W})
    @printf("predict: ")
    @time for i in 1:size(Y, 1)
        predict!(g, w, X, sub(y_hat, (i, 1:num_labels(g))), i)
    end
    @printf("losses: ")
    loss = log_loss(Y, y_hat)
    micro_f1 = micro_f1_calculate(y_hat, Y)
    accuracy = accuracy_calculate(y_hat, Y)
    return loss, micro_f1, accuracy
end

function learn{T}(g::StochasticGradientDescent{T}, w, X, Y, testX, testY; epochs=100, modn=10)
    y_hat = zeros(T, (size(Y, 1), num_labels(g)))
    test_y_hat = zeros(T, (size(testY, 1), num_labels(g)))

    for e in 1:epochs
        @printf("New epoch: %s\n", e)
        @time for i in 1:size(X, 1)
            if ((modn == 1) || (i % modn == 1))
                if savefile != ""
                    f = open(savefile, "a")
                    @printf(f, "Epoch %i Iter %i Weights", e, i)
                    for j in 1:length(w)
                        @printf(f, " %f", w[j])
                    end
                    @printf(f, "\n")
                    close(f)
                else
                    #@time loss, micro_f1, accuracy = dataset_log_loss(g, w, X, Y, y_hat)
    	            @time test_loss, test_micro, test_accuracy = dataset_log_loss(g, w, testX, testY, test_y_hat)
                    @printf("Epoch %i Iter %i (loss %4f): %s", e, i, test_loss, w[1:3]')
                    #@printf("\t train:Micro_F1: %4f,  Hamming Loss: %4f", micro_f1, 1.0 - accuracy)
                    @printf("\t test:Micro_F1: %4f,  Hamming Loss: %4f\n", test_micro, 1.0 - test_accuracy)
                end
            end
            if showtime
                @time train_samples!(g, w, X, Y, i:i, e, dropout)
            else
                train_samples!(g, w, X, Y, i:i, e, dropout)
            end
        end
    end
end

function whiten{T<:Number}(a::Array{T, 2})
    m = mean(a, 1)
    a = broadcast(-, a, m)
    s = std(a, 1)
    # Do not divide by 0 stddev or we will get NaN!
    for i in 1:length(s)
        if s[i] == 0.0
            s[i] = 1.0
        end
    end
    a = broadcast(/, a, s)
    return a, m, s
end

function whiten{T<:Number}(a::Array{T, 2}, m::Matrix{T}, s::Matrix{T})
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
regularization_constant = parsed_args["regularization"]
interval = parsed_args["interval"]
showtime = parsed_args["time"]
dropout = parsed_args["dropout"]

function flatten(s)
    s = replace(s, " ", "_")
    s = replace(s, ".", "_")
    s = replace(s, "-", "_")
    s = replace(s, ":", "_")
    s = replace(s, "'", "_")
    s = replace(s, "\"", "_")
    s = replace(s, "/", "_")
    s = replace(s, "\\", "_")
    return s
end

file_prefix = parsed_args["file_prefix"]
savefile = ""
if file_prefix != ""
    savefile = string(file_prefix, "_", flatten(join(ARGS, "_")))
end
if isfile(savefile)
   error("File already exists, please delete or move the file, or change the parameters: ", savefile)
end

#########################
# Read and cleanup data
#########################
train_features, train_labels = read_data(dataset, "train")
println("Successfully read data, now whitening...")
train_features, train_mean, train_std = whiten(train_features)
train_features = prepend_intercept(train_features)

test_features, test_labels = read_data(dataset, "test")
test_features = whiten(test_features, train_mean, train_std)
test_features = prepend_intercept(test_features)

# attempting sparsification
# train_features = sparse(train_features)
# test_features = sparse(test_features)

dimensions = size(train_features, 2)
nlabels = size(train_labels, 2)
@assert size(train_labels, 1) == size(train_features, 1)

#########################
# Setup Data and Run
#########################

RUNT = Float64

if adagrad
    @printf("SLN MLL AdaGrad\n")
    sln = SLN_MLL(RUNT, dimensions, nlabels, hidden_nodes)
    mweights = ones(length(sln.input_output) + length(sln.input_hidden) + length(sln.hidden_output))
    flat_weights!(sln, mweights)
    slnmllada = MultilabelSLNAdaGrad{RUNT}(zeros(RUNT, length(mweights)), nlabels, initial_learning_rate, sln, ones(RUNT, length(mweights)), SLN_MLL_Activation(sln), SLN_MLL_Deltas(sln), SLN_MLL_Derivatives(sln), regularization_constant)

    @time learn(slnmllada, mweights, train_features, train_labels, test_features, test_labels, epochs=nepochs, modn=interval)
else
    @printf("SLN MLL SGD\n")
    sln = SLN_MLL(RUNT, dimensions, nlabels, hidden_nodes)
    mweights = ones(length(sln.input_output) + length(sln.input_hidden) + length(sln.hidden_output))
    flat_weights!(sln, mweights)
    slnmllsgd = MultilabelSLNSGD{RUNT}(zeros(RUNT, length(mweights)), nlabels, initial_learning_rate, sln, SLN_MLL_Activation(sln), SLN_MLL_Deltas(sln), SLN_MLL_Derivatives(sln), regularization_constant)

    @time learn(slnmllsgd, mweights, train_features, train_labels, test_features, test_labels, epochs=nepochs, modn=interval)
end
