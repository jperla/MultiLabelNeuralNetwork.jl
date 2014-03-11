include("abstract_neural_networks.jl")

function zero!(nn::NeuralNetworkStorage)
    # Zeroes out all of the arrays in the neural network type
    names = names(nn)
    types = nn.types
    for t,n in zip(types, names)
        if isa(t, Weights)
        fill!(nn.(n), 0)
        end
    end
end

