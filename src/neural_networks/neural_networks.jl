include("abstract_neural_networks.jl")

function zero!(nn::NeuralNetworkStorage)
    # Zeroes out all of the Weights in the neural network
    fields = names(nn)
    types = typeof(nn).types
    for (t,f) in zip(types, fields)
        if t <: Weights
            fill!(nn.(f), 0)
        end
    end
end



function sigmoid_prime(x)
    return sigmoid(x) * (1 - sigmoid(x)
end
