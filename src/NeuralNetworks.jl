include("neural_networks/abstract_neural_networks.jl")

module NeuralNetworks

importall Base

export SLN_MLL, forward_propagate, gradient

include("neural_networks/sln_mll.jl")

end # module NeuralNetworks
