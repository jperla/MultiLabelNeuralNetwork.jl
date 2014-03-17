include("neural_networks/abstract_neural_networks.jl")

module NeuralNetworks

importall Base


export SLN_MLL, SLN_MLL_Activation, SLN_MLL_Deltas, SLN_MLL_Derivatives,
       VectorOrSubArrayVector,
       forward_propagate!, calculate_label_probabilities!,
       back_propagate!, sigmoid_prime, square_loss, log_loss,
       log_loss_prime, sigmoid, relu,
       zero!, gradientm,
       top_features, top_weights,
       hidden_nodes_table,
       assert_not_NaN,
       fill!, flat_weights!, read_data

include("neural_networks/sln_mll.jl")
include("neural_networks/neural_networks.jl")
include("neural_networks/loss_functions.jl")
include("neural_networks/read_data.jl")

end # module NeuralNetworks
