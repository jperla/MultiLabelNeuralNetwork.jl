include("neural_networks/abstract_neural_networks.jl")

module NeuralNetworks

importall Base

export SLN_MLL, forward_propagate, zero!, gradientm, log_loss

include("neural_networks/sln_mll.jl")
include("neural_networks/neural_networks.jl")
include("neural_networks/loss_functions.jl")

end # module NeuralNetworks
