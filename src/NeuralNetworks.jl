include("neural_networks/abstract_neural_networks.jl")

module NeuralNetworks

importall Base

export SLN_MLL, SLN_MLL_Activation,
       forward_propagate!, back_propagate!, sigmoid_prime, square_loss, log_loss, log_loss_prime, sigmoid_prime, calculate_label_probabilities,
       zero!, gradientm, log_loss

include("neural_networks/sln_mll.jl")
include("neural_networks/neural_networks.jl")
include("neural_networks/loss_functions.jl")

end # module NeuralNetworks
