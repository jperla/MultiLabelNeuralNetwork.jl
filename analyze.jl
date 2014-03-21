#!/usr/bin/env julia
import StochasticGradient: read_weights, calculate_losses
import NeuralNetworks: read_data, whiten, prepend_intercept, flat_weights_length
import MultilabelNeuralNetwork: SLN_MLL, MultilabelSLNSGD, flat_weights!

dataset = ARGS[1]
hidden_nodes = int64(ARGS[2])
weights_file = ARGS[3]
train_or_test = ARGS[4]

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

RUNT = Float64

sln = SLN_MLL(RUNT, dimensions, nlabels, hidden_nodes)    
mweights = zeros(RUNT, flat_weights_length(sln))
flat_weights!(sln, mweights)
classifier = MultilabelSLNSGD(sln, mweights)

losses = calculate_losses(classifier, weights, X, Y)
@printf(" Log Loss | Hamming | Micro F1 |   0-1 \n")
for i in 1:size(losses, 1)
    @printf("  %0.4f    %0.4f    %0.4f    %0.4f\n", losses[i, 1], losses[i, 2], losses[i, 3], losses[i, 4])
end

