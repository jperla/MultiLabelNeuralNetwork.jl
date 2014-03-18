using Base.Test

import Calculus: check_gradient

import NeuralNetworks: SLN_MLL, forward_propagate!, back_propagate!,
                       fill!, flat_weights!, assert_not_NaN,
                       gradient, calculate_label_probabilities!, log_loss, square_loss, read_data

sln = SLN_MLL(10, 3, 2)

@test assert_not_NaN(sln)
x = sln.input_output[1]
sln.input_output[1] = NaN
#@test !assert_not_NaN(sln)
sln.input_output[1] = x

# test weights flattening/filling/reading/writing
flat1 = zeros(Float64, length(sln.input_hidden) + length(sln.input_output) + length(sln.hidden_output))
flat_weights!(sln, flat1)
fill!(sln, flat1)
flat2 = zeros(Float64, length(flat1))
flat_weights!(sln, flat2)
@test flat1 == flat2
sln2 = SLN_MLL(10, 3, 2)
flat3 = zeros(Float64, length(flat1))
flat_weights!(sln2, flat3)
@test flat2 != flat3
fill!(sln2, flat2)
flat4 = zeros(Float64, length(flat1))
flat_weights!(sln2, flat4)
@test flat2 == flat4

"""
x0 = [1 0 0 1 0.5 1.0 0.0 1.0 0.0 1.0;]
q(x) = (calculate_label_probabilities(sln, x)[1], gradient(sln, x)[1,:])

println(calculate_label_probabilities(sln, x0))

function f(weights)
    fill!(sln, weights)
    return calculate_label_probabilities(sln, x0)
end

function g(weights)
    fill!(sln, weights)
    return gradient(sln, x0)
end

# TODO: Check backprop
y = [1.0,0, 0]
derivatives = back_propagate!(sln, x0, y)
@printf("Derivatives: %s", derivatives)


y = calculate_label_probabilities(sln, x0)
derivatives = back_propagate!(sln, x0, y)
@printf("Derivatives with no error... %s", derivatives)
"""

# Cap log loss
@test 100 < log_loss(1.0, 0.0) < Inf


#@test log_loss(1.0,.99999999) == 0.0

features, labels = read_data("scene", "test")
@test size(features) == (1196, 294)
@test size(labels) == (1196, 6)

features, labels = read_data("scene", "train")
@test size(features) == (1211, 294)
@test size(labels) == (1211, 6)


@test 0 < log_loss(1.0,.9999) < .1
@test square_loss(0.0, 0.0) == 0

