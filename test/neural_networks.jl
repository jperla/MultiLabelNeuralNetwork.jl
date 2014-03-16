using Base.Test

import Calculus: check_gradient

import NeuralNetworks: SLN_MLL, forward_propagate!,
                       fill!, flat_weights,
                       gradient, calculate_label_probabilities, log_loss, read_data

sln = SLN_MLL(10, 3, 2)

# test weights flattening/filling/reading/writing
flat1 = flat_weights(sln)
fill!(sln, flat1)
flat2 = flat_weights(sln)
@test flat1 == flat2
sln2 = SLN_MLL(10, 3, 2)
@test flat2 != flat_weights(sln2)
fill!(sln2, flat2)
@test flat2 == flat_weights(sln2)

x0 = [1, 0, 0, 1, 0.5, 1.0, 0.0, 1.0, 0.0, 1.0]
q(x) = (calculate_label_probabilities(sln, x)[1], gradient(sln, x)[1,:])

function f(weights)
    fill!(sln, weights)
    return calculate_label_probabilities(sln, x0)
end

function g(weights)
    fill!(sln, weights)
    return gradient(sln, x0)
end

# TODO: Check backprop
y = [1.0,0,0,1,1,0,1,1,0,0]
derivatives = backpropagate!(sln, x, y)
println("Derivatives: $derivatives")

#@test check_gradient(f, g, flat_weights(sln)) < 1e-6

@test log_loss(1.0,0.0) == Inf


#@test log_loss(1.0,.99999999) == 0.0

features, labels = read_data("scene", "test")
@test size(features) == (1196, 294)
@test size(labels) == (1196, 6)

features, labels = read_data("scene", "train")
@test size(features) == (1211, 294)
@test size(labels) == (1211, 6)


@test 0 < log_loss(1.0,.9999) < .1
@test square_loss(0) == 0
