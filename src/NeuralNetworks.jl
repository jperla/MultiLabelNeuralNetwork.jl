include("neural_networks/abstract_neural_networks.jl")

module NeuralNetworks

importall Base


export SLN_MLL, SLN_MLL_Activation,
       forward_propagate!, calculate_label_probabilities,
       back_propagate!, sigmoid_prime, square_loss, log_loss,
       log_loss_prime, sigmoid_prime,
       zero!, gradientm,
       top_features, top_weights,
       hidden_nodes_table

include("neural_networks/sln_mll.jl")
include("neural_networks/neural_networks.jl")
include("neural_networks/loss_functions.jl")

end # module NeuralNetworks
