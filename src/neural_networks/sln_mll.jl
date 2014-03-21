include("neural_networks.jl")

#####################################
# Weights for the neural network
#####################################

type SLN_MLL{T} <: NeuralNetworkStorage{T}
    # single layer neural network for multi label learning with skip level weights
    input_hidden::WeightMatrix{T} # weights to calculate hidden layer
    hidden_output::WeightMatrix{T} # weights to the final layer
    input_output::WeightMatrix{T} # skip-level weights direction from input to outputs
end

type SLN_MLL_Activation{T} <: NeuralNetworkStorage{T}
    # single layer neural network activation levels after being trained on input
    hidden::Activations{T}
    output::Activations{T}
end

type SLN_MLL_Deltas{T} <: NeuralNetworkStorage{T}
    hidden::Array{T}
    output::Array{T}
end

type SLN_MLL_Derivatives{T} <: NeuralNetworkStorage{T}
    input_hidden::Array{T, 2} # derivatives of weights to calculate hidden layer
    hidden_output::Array{T, 2} # derivatives of weights to the final layer
    input_output::Array{T, 2} # skip-level weights direction from input to outputs
end

#####################################
# Constructors
#####################################

function SLN_MLL(T, num_dimensions::Int, num_labels::Int, num_hidden::Int)
    input_output = convert(Array{T, 2}, randn(num_dimensions, num_labels))
    input_hidden = convert(Array{T, 2}, randn(num_dimensions, num_hidden))
    hidden_output = convert(Array{T, 2}, randn(num_hidden, num_labels))
    SLN_MLL{T}(input_hidden, hidden_output, input_output)
end

function SLN_MLL_Activation{T}(sln::SLN_MLL{T})
    SLN_MLL_Activation{T}(zeros(num_hidden(sln)), zeros(num_output(sln)))
end

function SLN_MLL_Deltas{T}(sln::SLN_MLL{T})
    SLN_MLL_Deltas{T}(zeros(num_hidden(sln)), zeros(num_output(sln)))
end

function SLN_MLL_Derivatives{T}(sln::SLN_MLL{T})
    input_hidden = zeros(size(sln.input_hidden))
    input_output = zeros(size(sln.input_output))
    hidden_output = zeros(size(sln.hidden_output))
    SLN_MLL_Derivatives{T}(input_hidden, hidden_output, input_output)
end

#####################################
# Size helper functions
#####################################

num_dimensions(sln::SLN_MLL) = size(sln.input_output, 1)
num_hidden(sln::SLN_MLL) = size(sln.input_hidden, 2)
num_output(sln::SLN_MLL) = size(sln.input_output, 2)
num_labels(sln::SLN_MLL) = num_output(sln)

#####################################
# Weight read/write helper functions
#####################################

function fill!{T}(sln::SLN_MLL{T}, weights::WeightVector{T})
    io = length(sln.input_output)
    ih = length(sln.input_hidden)
    ho = length(sln.hidden_output)
    @assert (io + ih + ho) == length(weights)
    for i in 1:io
        sln.input_output[i] = weights[i]
    end
    for i in 1:ih
        sln.input_hidden[i] = weights[io+i]
    end
    for i in 1:ho
        sln.hidden_output[i] = weights[io+ih+i]
    end
end

function flat_weights!{T}(sln::Union(SLN_MLL{T}, SLN_MLL_Derivatives), weights::Array{T})
    io = length(sln.input_output)
    ih = length(sln.input_hidden)
    ho = length(sln.hidden_output)
    for i in 1:io
        weights[i] = sln.input_output[i]
    end
    for i in 1:ih
        weights[io+i] = sln.input_hidden[i]
    end
    for i in 1:ho
        weights[io+ih+i] = sln.hidden_output[i]
    end
end

#####################################
# Classification / Testing
#####################################

function forward_propagate!{T,U<:FloatingPoint}(sln::SLN_MLL{T}, activation::SLN_MLL_Activation{T}, X::Matrix{U}, i::Int, dropout::Bool)
    @assert size(X, 2) == num_dimensions(sln) == size(sln.input_hidden, 1)

    for k in 1:size(sln.input_hidden, 2)
        h = 0.0
        for j in 1:size(X, 2)
            h += X[i, j] * sln.input_hidden[j, k]
        end

        # dropout only 25% of time
        if dropout && randbool()
            activation.hidden[k] = 0
        else
            activation.hidden[k] = standard_link(h)
        end
    end

    @assert assert_not_NaN(activation.hidden)

    for k in 1:size(sln.input_output, 2)
        h = 0.0
        for j in 1:size(X, 2)
            h += X[i, j] * sln.input_output[j, k]
        end
        activation.output[k] = h
    end

    @assert assert_not_NaN(sln.hidden_output)
    for k in 1:size(sln.input_output, 2)
        h = 0.0
        for j in 1:length(activation.hidden)
            h += activation.hidden[j] * sln.hidden_output[j, k]
        end
        activation.output[k] += h
    end

    @assert length(activation.output) == num_labels(sln)
    @assert assert_not_NaN(activation.output)

end


# function forward_propagate!{T,U<:FloatingPoint}(sln::SLN_MLL{T}, activation::SLN_MLL_Activation{T}, X::SparseMatrixCSC{U,Int64}, i::Int, dropout::Bool)
#     @assert size(X, 2) == num_dimensions(sln) == size(sln.input_hidden, 1)

#     for k in 1:size(sln.input_hidden, 2)
#         h = X[i,:] * sln.input_hidden[:, k]
#         activation.hidden[k] = relu(h[1,1])

#         if dropout && randbool()
#             activation.hidden[k] = 0
#         else
# 	    debug1 = typeof(activation.hidden[k])
# 	    debug2 = typeof(relu(h))
#             println("Error assigning $debug2 to $debug1")
#             activation.hidden[k] = relu(h[1,1])
#         end
#     end
#     @assert assert_not_NaN(activation.hidden)

#     for k in 1:size(sln.input_output, 2)
#         h = X[i, :] * sln.input_output[:, k]
#         activation.output[k] = h[1,1]
#     end

#     @assert assert_not_NaN(sln.hidden_output)
#     for k in 1:size(sln.input_output, 2)
#         h = activation.hidden' * sln.hidden_output[:, k][:]
#         activation.output[k] += h[1,1]
#     end

#     @assert length(activation.output) == num_labels(sln)
#     @assert assert_not_NaN(activation.output)
# end

function calculate_label_probabilities!{T,U<:FloatingPoint,W<:FloatingPoint}(sln::SLN_MLL{T}, X::AbstractMatrix{U}, y_hat::AbstractArray{W}, i::Int, dropout::Bool)
    activation = SLN_MLL_Activation(sln)
    forward_propagate!(sln, activation, X, i, dropout)
    @assert length(y_hat) == length(activation.output)
    for j in 1:length(y_hat)
        y_hat[j] = sigmoid(activation.output[j])
    end

end

#####################################
# Training
#####################################

function back_propagate!{T,U<:FloatingPoint,W<:FloatingPoint}(sln::SLN_MLL{T}, activation, deltas, derivatives, X::AbstractMatrix{U}, Y::AbstractMatrix{W}, i::Int, dropout::Bool)
    # Calculates the derivatives of all weights in the neural network through backpropagation
    ################################################################
    #   Calculate delta_k
    #   Calculate delta_j for each interior node
    #   Calculate weight updates
    ################################################################
    forward_propagate!(sln, activation, X, i, dropout)
    @assert assert_not_NaN(activation)

    for j=1:size(Y, 2)
        deltas.output[j] = log_loss_prime(Y[i,j], sigmoid(activation.output[j])) * sigmoid_prime(activation.output[j])
        if isequal(deltas.output[j], NaN)
            logresult = log_loss_prime(Y[i,j], sigmoid(activation.output[j]))
            sigresult = sigmoid_prime(activation.output[j])
            println("NaN spotted: Delta of output #$j, logprim:$logresult[1:3]..., sigprime:$sigresult[1:3]...")
        end
    end

    for j = 1:length(deltas.hidden)
        deltas.hidden[j] = 0
        for k = 1:length(activation.output)
            deltas.hidden[j] += deltas.output[k] * sln.hidden_output[j,k]
        end
    end

    @assert assert_not_NaN(deltas)
    @assert assert_not_NaN(derivatives)
    calculate_derivatives!(sln, activation, derivatives, deltas, X, i)
    @assert assert_not_NaN(derivatives)
end


function calculate_derivatives!{T,U<:FloatingPoint}(sln::SLN_MLL{T}, activation::SLN_MLL_Activation{T}, derivatives::SLN_MLL_Derivatives{T}, deltas::SLN_MLL_Deltas{T}, X::AbstractMatrix{U}, i::Int)
    ############################################################
    #  calculate derivatives for weights from input to hidden layer
    ############################################################

    @assert length(deltas.hidden) == size(derivatives.input_hidden, 2)
    for j = 1:size(derivatives.input_hidden, 1)
        for k = 1:size(derivatives.input_hidden, 2)
            derivatives.input_hidden[j,k] = deltas.hidden[k] * X[i,j]
 	    if isequal(derivatives.input_hidden[j,k], NaN)
		deltaj = deltas.hidden[k]
	        println("NaN Spotted: i: $i, j: $j, X[i,j] = $X[i,j], deltas.hidden: $deltaj")
	    end
        end
    end

    @assert assert_not_NaN(derivatives)

    for j = 1:size(derivatives.hidden_output, 1)
        for k = 1:size(derivatives.hidden_output, 2)
            derivatives.hidden_output[j,k] = deltas.output[k] * activation.hidden[j]
        end
    end

    for j = 1:size(derivatives.input_output, 1)
        for k = 1:size(derivatives.input_output, 2)
            derivatives.input_output[j,k] = deltas.output[k] * X[i,j]
        end
    end

    return derivatives
end

###########################################
# Inspection, Debugging, and Documentation
###########################################

function hidden_nodes_table{T<:String}(io, sln::SLN_MLL,
                                       input_names::Vector{T}, output_names::Vector{T},
                                       N::Int=6)
    # Prints the top tags and top words for every hidden node in the neural network
    @printf(io, "\\usepackage{multirow}\n")
    @printf(io, "\\begin{tabular}{ |l|l|l| }\n")
    @printf(io, "\\hline\n")
    @printf(io, "\\multicolumn{3}{ |c| }{Hidden Units} \\\\\n")
    @printf(io, "\\hline\n")
    @printf(io, "Hidden Unit    &    Labels     &     Words\n")
    for h in 1:num_hidden(sln)
        @printf(io, "\\multirow{3}{*}{%s}\n", h)
        labels = [(output_names[i], w) for (i, w) in top_weights(sln.hidden_output[h,:][:])]
        features = top_features(input_names, sln.input_hidden[:,h])
        for i in 1:min(N, max(length(features), length(labels)))
            ls = if i <= length(labels) @sprintf("%s (%.4f)", labels[i][1], labels[i][2]) else "" end
            fs = if i < length(features) @sprintf("%s (%.4f)", features[i][1], features[i][2]) else "" end
            @printf(io, " & %35s & %35s \\\\\n", ls, fs)
        end
        @printf(io, "\\hline\n")
    end
    @printf(io, "\\end{tabular}\n")
end

