module NeuralNetworks

include("neural_networks/abstract_neural_networks.jl")

importall Base


export SLN_MLL, SLN_MLL_Activation, SLN_MLL_Deltas, SLN_MLL_Derivatives,
       num_labels,
       forward_propagate!, calculate_label_probabilities!,
       back_propagate!, sigmoid_prime, square_loss, log_loss, zero_one_loss,
       log_loss_prime, sigmoid, relu, standard_link,
       zero!, gradientm,
       top_features, top_weights,
       hidden_nodes_table,
       assert_not_NaN,
       WeightMatrix, WeightVector,
       fill!, flat_weights!, flat_weights_length,
       read_data

include("neural_networks/sln_mll.jl")
include("neural_networks/neural_networks.jl")
include("neural_networks/loss_functions.jl")
include("neural_networks/read_data.jl")

end # module NeuralNetworks
