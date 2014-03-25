#!/usr/bin/env julia
import StochasticGradient: train_samples!, StochasticGradientDescent
import NeuralNetworks: SLN_MLL, SLN_MLL_Activation, SLN_MLL_Deltas, SLN_MLL_Derivatives,
                       read_data, flat_weights!, flat_weights_length,
                       log_loss, assert_not_NaN,
                       whiten, prepend_intercept,
                       TanhLinkFunction, RectifiedLinearUnitLinkFunction, SigmoidLinkFunction
import MultilabelNeuralNetwork: MultilabelSLN, MultilabelSLNAdaGrad, MultilabelSLNSGD,
                                predict!, calculate_gradient!
import Thresholds: accuracy_calculate, micro_f1_calculate

require("args.jl")

function num_labels(g)
    g.num_labels
end

function dataset_log_loss{T,U<:FloatingPoint,W<:FloatingPoint}(g, w::Vector{T},
                                                               X::AbstractMatrix{U}, Y::AbstractMatrix{W},
                                                               y_hat::AbstractMatrix{W})
    @printf("predict: ")
    @time predict!(g, w, X, y_hat)
    @printf("losses: ")
    loss = log_loss(Y, y_hat)
    micro_f1 = micro_f1_calculate(y_hat, Y)
    accuracy = accuracy_calculate(y_hat, Y)
    return loss, micro_f1, accuracy
end

function learn{T}(g::StochasticGradientDescent{T}, w, X, Y, testX, testY; epochs=100, modn=10, showtime=false, savefile=false)
    y_hat = zeros(T, (size(Y, 1), num_labels(g)))
    test_y_hat = zeros(T, (size(testY, 1), num_labels(g)))

    for e in 1:epochs
        @printf("New epoch: %s\n", e)
        @time for i in 1:size(X, 1)
            if ((modn == 1) || (i % modn == 1))
                if savefile != ""
                    f = open(savefile, "a")
                    samples_seen = ((e - 1) * size(X, 1)) + i
                    @printf(f, "%i", samples_seen)
                    for j in 1:length(w)
                        @printf(f, " %4f", w[j])
                    end
                    @printf(f, "\n")
                    close(f)
                else
                    @time loss, micro_f1, accuracy = dataset_log_loss(g, w, X, Y, y_hat)
    	            @time test_loss, test_micro, test_accuracy = dataset_log_loss(g, w, testX, testY, test_y_hat)
                    @printf("Epoch %i Iter %i (loss %4f): %s", e, i, test_loss, w[1:3]')
                    @printf("\t train:Micro_F1: %4f,  Hamming Loss: %4f", micro_f1, 1.0 - accuracy)
                    @printf("\t test:Micro_F1: %4f,  Hamming Loss: %4f\n", test_micro, 1.0 - test_accuracy)
                end
            end
            if showtime
                @time train_samples!(g, w, X, Y, i:i, e)
            else
                train_samples!(g, w, X, Y, i:i, e)
            end
        end
    end
end

function main()
#########################
# Setup args
#########################

parsed_args = parse_commandline()
dataset = parsed_args["dataset"]

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

file_prefix = parsed_args["file"]
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
@assert dimensions == size(test_features, 2)
@assert nlabels == size(test_labels, 2)

#########################
# Setup Data and Run
#########################

RUNT = Float64

slnmll = slnmll_from_args(dimensions, nlabels, parsed_args)
mweights = zeros(eltype(slnmll.sln.input_hidden), flat_weights_length(slnmll.sln))
#flat_weights!(sln, mweights)

nepochs = parsed_args["epochs"]
interval = parsed_args["interval"]
showtime = parsed_args["time"]

@time learn(slnmll, mweights, train_features, train_labels, test_features, test_labels, epochs=nepochs, modn=interval, showtime=showtime, savefile=savefile)

end

main()
