#!/usr/bin/env julia
import StochasticGradient: read_weights, calculate_losses
import NeuralNetworks: read_data, whiten, prepend_intercept, flat_weights_length
import MultilabelNeuralNetwork: SLN_MLL, MultilabelSLNSGD, flat_weights!

require("args.jl")

parsed_args = parse_commandline()
dataset = parsed_args["dataset"]
hidden_nodes = parsed_args["hidden"]
weights_file = parsed_args["file"]
train_or_test = "train"

train_features, train_labels = read_data(dataset, "train")
println("Successfully read data, now whitening...")
train_features, train_mean, train_std = whiten(train_features)
train_features = prepend_intercept(train_features)

test_features, test_labels = read_data(dataset, "test")
test_features = whiten(test_features, train_mean, train_std)
test_features = prepend_intercept(test_features)

dimensions = size(train_features, 2)
nlabels = size(train_labels, 2)
@assert size(train_labels, 1) == size(train_features, 1)

if train_or_test == "train"
    X, Y = train_features, train_labels
else
    X, Y = test_features, test_labels
end

iter, weights = read_weights(weights_file)

classifier = slnmll_from_args(dimensions, nlabels, parsed_args)

losses = calculate_losses(classifier, weights, X, Y)
@printf(" Log Loss | Hamming | Micro F1 |   0-1 \n")
for i in 1:size(losses, 1)
    @printf("  %0.4f    %0.4f    %0.4f    %0.4f\n", losses[i, 1], losses[i, 2], losses[i, 3], losses[i, 4])
end

