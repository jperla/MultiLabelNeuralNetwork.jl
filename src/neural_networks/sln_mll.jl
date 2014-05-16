include("neural_networks.jl")

#####################################
# Weights for the neural network
#####################################

immutable type SLN_MLL{T} <: NeuralNetworkStorage{T}
    # single layer neural network for multi label learning with skip level weights
    input_hidden::WeightMatrix{T} # weights to calculate hidden layer
    hidden_output::WeightMatrix{T} # weights to the final layer
    input_output::WeightMatrix{T} # skip-level weights direction from input to outputs
    hidden_link::LinkFunction
    output_link::LinkFunction
end

type SLN_MLL_Activation{T} <: NeuralNetworkStorage{T}
    # single layer neural network activation levels after being trained on input
    hidden::Activations{T}
    hidden_linked::Activations{T} # after applying link function
    output::Activations{T}
    output_linked::Activations{T}
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

function SLN_MLL(T, num_dimensions::Int, num_labels::Int, num_hidden::Int, hidden_link::LinkFunction, output_link::LinkFunction)
    input_output = convert(Array{T, 2}, randn(num_dimensions, num_labels))
    input_hidden = convert(Array{T, 2}, randn(num_dimensions, num_hidden))
    hidden_output = convert(Array{T, 2}, randn(num_hidden, num_labels))
    SLN_MLL{T}(input_hidden, hidden_output, input_output, hidden_link, output_link)
end

function SLN_MLL_Activation{T}(sln::SLN_MLL{T})
    SLN_MLL_Activation{T}(zeros(num_hidden(sln)), zeros(num_hidden(sln)), zeros(num_output(sln)), zeros(num_output(sln)))
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

function flat_weights_length(sln::SLN_MLL)
    io = length(sln.input_output)
    ih = length(sln.input_hidden)
    ho = length(sln.hidden_output)
    return io + ih + ho
end

function fill!{T}(sln::SLN_MLL{T}, weights::AbstractArray{T})
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

function flat_weights!{T}(sln::Union(SLN_MLL{T}, SLN_MLL_Derivatives), weights::AbstractArray{T})
    io = length(sln.input_output)
    ih = length(sln.input_hidden)
    ho = length(sln.hidden_output)
    @assert length(weights) == io + ih + ho
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

function forward_propagate!{T,U<:FloatingPoint}(sln::SLN_MLL{T}, activation::SLN_MLL_Activation{T}, X::AbstractMatrix{U}, i::Int, dropout::Bool=false, testing::Bool=false)
    @assert size(X, 2) == num_dimensions(sln) == size(sln.input_hidden, 1)

    for k in 1:size(sln.input_hidden, 2)
        if (!testing) && dropout && randbool() # dropout only 50% of time
            activation.hidden[k] = inverse_link_function(0)
            activation.hidden_linked[k] = 0
        else
            h = 0.0
            for j in 1:size(X, 2)
                h += X[i, j] * sln.input_hidden[j, k]
            end
            
            activation.hidden[k] = h
            
            if dropout && testing
                activation.hidden_linked[k] = link_function(sln.hidden_link, h) * .5
            else
                activation.hidden_linked[k] = link_function(sln.hidden_link, h)
            end
        end
    end

    @assert assert_not_NaN(activation.hidden)
    @assert assert_not_NaN(activation.hidden_linked)

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
            h += activation.hidden_linked[j] * sln.hidden_output[j, k]
        end
        activation.output[k] += h
    end

    activation.output_linked = link_function(sln.output_link, activation.output) 
    
    @assert length(activation.output) == num_labels(sln)
    @assert assert_not_NaN(activation.output)
end

function forward_propagate!{T,U<:FloatingPoint}(sln::SLN_MLL{T}, activation::SLN_MLL_Activation{T}, X::SparseMatrixCSC{U,Int64}, i::Int, dropout::Bool=false, testing::Bool=false)
    @assert size(X, 2) == num_dimensions(sln) == size(sln.input_hidden, 1)
    
    # implement dropout later
    @assert dropout == false

    activation.hidden = (X[i,:] * sln.input_hidden)[:]
    activation.hidden_linked = link_function(sln.hidden_link, activation.hidden)
    @assert assert_not_NaN(activation.hidden)

    activation.output = (X[i,:] * sln.input_output)[:]
    @assert assert_not_NaN(activation.output)
    
    activation.output += (activation.hidden_linked' * sln.hidden_output)[:]

    activation.output_linked = link_function(sln.output_link, activation.output)
    @assert length(activation.output) == num_labels(sln)
    @assert assert_not_NaN(activation.output)
end

function calculate_label_probabilities!{T,U<:FloatingPoint,W<:FloatingPoint}(sln::SLN_MLL{T}, activation::SLN_MLL_Activation{T}, X::AbstractMatrix{U}, y_hat::AbstractArray{W}, i::Int)
    forward_propagate!(sln, activation, X, i, false, true)
    @assert length(y_hat) == length(activation.output)
    for j in 1:length(y_hat)
        y_hat[j] = activation.output_linked[j]
    end
end

#####################################
# Training
#####################################

function back_propagate!{T,U<:FloatingPoint,W<:FloatingPoint}(sln::SLN_MLL{T}, activation, deltas, derivatives, X::AbstractMatrix{U}, Y::AbstractMatrix{W}, i::Int, dropout::Bool=false)
    # Calculates the derivatives of all weights in the neural network through backpropagation
    forward_propagate!(sln, activation, X, i, dropout, false)
    @assert assert_not_NaN(activation)
    num_output = size(Y, 2)
    
    for k=1:num_output
        deltas.output[k] = log_loss_prime(Y[i,k], activation.output_linked[k]) * link_function_prime(sln.output_link, activation.output[k])
    end
    
    for j = 1:length(deltas.hidden)
        deltas.hidden[j] = 0
        for k = 1:length(activation.output)
            deltas.hidden[j] += deltas.output[k] * sln.hidden_output[j,k]
        end
        deltas.hidden[j] *= link_function_prime(sln.hidden_link, activation.hidden[j])
    end

    @assert assert_not_NaN(deltas)
    calculate_derivatives!(sln, activation, derivatives, deltas, X, i)
    @assert assert_not_NaN(derivatives)
end

function calculate_derivatives!{T,U<:FloatingPoint}(sln::SLN_MLL{T}, activation::SLN_MLL_Activation{T}, derivatives::SLN_MLL_Derivatives{T}, deltas::SLN_MLL_Deltas{T}, X::AbstractMatrix{U}, i::Int)
    @assert length(deltas.hidden) == size(derivatives.input_hidden, 2)
    
    derivatives.input_hidden = X[i,:]' * deltas.hidden'

    @assert assert_not_NaN(derivatives)

    for j = 1:size(derivatives.hidden_output, 1)
        for k = 1:size(derivatives.hidden_output, 2)
            derivatives.hidden_output[j,k] = deltas.output[k] * activation.hidden_linked[j]
        end
    end
    
    derivatives.input_output = X[i,:]' * deltas.output'
    
    @assert assert_not_NaN(derivatives)

    return derivatives
end


function old_back_propagate!{T,U<:FloatingPoint,W<:FloatingPoint}(sln::SLN_MLL{T}, activation, deltas, derivatives, X::AbstractMatrix{U}, Y::AbstractMatrix{W}, i::Int, dropout::Bool=false)
    # Calculates the derivatives of all weights in the neural network through backpropagation
    forward_propagate!(sln, activation, X, i, dropout, false)
    @assert assert_not_NaN(activation)
    num_output = size(Y, 2)
    
    for k=1:num_output
        deltas.output[k] = log_loss_prime(Y[i,k], activation.output_linked[k]) * link_function_prime(sln.output_link, activation.output[k])
    end
    
    for j = 1:length(deltas.hidden)
        deltas.hidden[j] = 0
        for k = 1:length(activation.output)
            deltas.hidden[j] += deltas.output[k] * sln.hidden_output[j,k]
        end
        deltas.hidden[j] *= link_function_prime(sln.hidden_link, activation.hidden[j])
    end

    @assert assert_not_NaN(deltas)
    old_calculate_derivatives!(sln, activation, derivatives, deltas, X, i)
    @assert assert_not_NaN(derivatives)
end

function old_calculate_derivatives!{T,U<:FloatingPoint}(sln::SLN_MLL{T}, activation::SLN_MLL_Activation{T}, derivatives::SLN_MLL_Derivatives{T}, deltas::SLN_MLL_Deltas{T}, X::AbstractMatrix{U}, i::Int)
    @assert length(deltas.hidden) == size(derivatives.input_hidden, 2)
    for j = 1:size(derivatives.input_hidden, 1)
        for k = 1:size(derivatives.input_hidden, 2)
            derivatives.input_hidden[j,k] = deltas.hidden[k] * X[i,j]
        end
    end

    @assert assert_not_NaN(derivatives)

    for j = 1:size(derivatives.hidden_output, 1)
        for k = 1:size(derivatives.hidden_output, 2)
            derivatives.hidden_output[j,k] = deltas.output[k] * activation.hidden_linked[j]
        end
    end

    for j = 1:size(derivatives.input_output, 1)
        for k = 1:size(derivatives.input_output, 2)
            derivatives.input_output[j,k] = deltas.output[k] * X[i,j]
        end
    end    
    
    @assert assert_not_NaN(derivatives)

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

