include("abstract_neural_networks.jl")
include("neural_networks.jl")

#####################################
# Weights for the neural network
#####################################

type SLN_MLL <: NeuralNetworkStorage
    # single layer neural network for multi label learning with skip level weights
    input_hidden::Weights # weights to calculate hidden layer
    hidden_output::Weights # weights to the final layer
    input_output::Weights # skip-level weights direction from input to outputs
end

type SLN_MLL_Activation <: NeuralNetworkStorage
    # single layer neural network activation levels after being trained on input
    hidden::Activations
    output::Activations
end


type SLN_MLL_Deltas <: NeuralNetworkStorage
    hidden::Array{Float64}
    output::Array{Float64}
end

type SLN_MLL_Derivatives <: NeuralNetworkStorage
    input_hidden::Array{Float64,2} # derivatives of weights to calculate hidden layer
    hidden_output::Array{Float64,2} # derivatives of weights to the final layer
    input_output::Array{Float64,2} # skip-level weights direction from input to outputs
end

#####################################
# Constructors
#####################################

function SLN_MLL(num_dimensions::Int, num_labels::Int, num_hidden::Int)
    input_output = randn(num_dimensions, num_labels)
    input_hidden = randn(num_dimensions, num_hidden)
    hidden_output = randn(num_hidden, num_labels)
    SLN_MLL(input_hidden, hidden_output, input_output)
end

function SLN_MLL_Activation(sln::SLN_MLL)
    SLN_MLL_Activation(zeros(num_hidden(sln)), zeros(num_output(sln)))
end

function SLN_MLL_Deltas(sln::SLN_MLL)
    SLN_MLL_Deltas(zeros(num_hidden(sln)), zeros(num_output(sln)))
end

function SLN_MLL_Derivatives(sln::SLN_MLL)
    input_hidden = zeros(size(sln.input_hidden))
    input_output = zeros(size(sln.input_output))
    hidden_output = zeros(size(sln.hidden_output))
    SLN_MLL_Derivatives(input_hidden, hidden_output, input_output)
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

function fill!(sln::SLN_MLL, weights::Vector{Weight})
    io = length(sln.input_output)
    ih = length(sln.input_hidden)
    ho = length(sln.hidden_output)
    @assert (io + ih + ho) == length(weights)
    sln.input_output[1:end] = weights[1:io]
    sln.input_hidden[1:end] = weights[io+1:io+ih]
    sln.hidden_output[1:end] = weights[io+ih+1:end]
end

function flat_weights(sln::Union(SLN_MLL, SLN_MLL_Derivatives))
    io = length(sln.input_output)
    ih = length(sln.input_hidden)
    ho = length(sln.hidden_output)
    weights = ones(io + ih + ho)
    weights[1:io] = sln.input_output[1:end]
    weights[io+1:io+ih] = sln.input_hidden[1:end]
    weights[io+ih+1:end] = sln.hidden_output[1:end]
    return weights
end

#####################################
# Classification / Testing
#####################################

function forward_propagate!(sln::SLN_MLL, activation::SLN_MLL_Activation, x::Sample)
    @assert length(x) == num_dimensions(sln)
    activation.hidden = relu(x' * sln.input_hidden)[:]
    @assert assert_not_NaN(activation.hidden)
    activation_input_out = (x' * sln.input_output)[:]
    @assert assert_not_NaN(sln.hidden_output)
    activation_hidden_out = (activation.hidden' * sln.hidden_output)[:]
    activation.output =  activation_hidden_out .+ activation_input_out
    @assert assert_not_NaN(activation_input_out)
    @assert assert_not_NaN(activation_hidden_out)
    @assert length(activation.output) == num_labels(sln)
    @assert assert_not_NaN(activation.output)
    #return activation
end

function calculate_label_probabilities(sln::SLN_MLL, x::Sample)
    activation = SLN_MLL_Activation(sln)
    forward_propagate!(sln, activation, x)
    return sigmoid(activation.output)[:]
end

#####################################
# Training
#####################################

function back_propagate!(sln::SLN_MLL, activation, deltas, derivatives, x::Sample, y::Labels)
    # Calculates the derivatives of all weights in the neural network through backpropagation
    ################################################################
    #   Calculate delta_k
    #   Calculate delta_j for each interior node
    #   Calculate weight updates
    ################################################################
    #activation = SLN_MLL_Activation(sln)
    forward_propagate!(sln, activation, x)
    probabilities = sigmoid(activation.output)[:]
    @assert assert_not_NaN(activation)

    #deltas = SLN_MLL_Deltas(sln)
    for i=1:length(y)
        deltas.output[i] = log_loss_prime(y[i],probabilities[i]) * sigmoid_prime(activation.output[i])
	if isequal(deltas.output[i], NaN)
	    logresult = log_loss_prime(y[i],probabilities[i])
	    sigresult = sigmoid_prime(activation.output[i]) 
	    println("NaN spotted: Delta of output #$i, logprim:$logresult, sigprime:$sigresult")
	end
    end

    for i = 1:length(deltas.hidden)
        for k = 1:length(activation.output)
            deltas.hidden[i] += deltas.output[k] * sln.hidden_output[i,k]
        end
    end

    @assert assert_not_NaN(deltas)

    #derivatives = SLN_MLL_Derivatives(sln)
    @assert assert_not_NaN(derivatives)
    calculate_derivatives!(sln, activation, derivatives, deltas, x)
    @assert assert_not_NaN(derivatives)
    #return derivatives
end


function calculate_derivatives!(sln::SLN_MLL, activation::SLN_MLL_Activation, derivatives::SLN_MLL_Derivatives, deltas::SLN_MLL_Deltas, x::Sample)
    ############################################################
    #  calculate derivatives for weights from input to hidden layer
    ############################################################

    @assert length(deltas.hidden) == size(derivatives.input_hidden, 2)
    for i = 1:size(derivatives.input_hidden, 1)
        for j = 1:size(derivatives.input_hidden, 2)
            derivatives.input_hidden[i,j] = deltas.hidden[j] * x[i]
 	    if isequal(derivatives.input_hidden[i,j], NaN)
		deltaj = deltas.hidden[j]
	        println("NaN Spotted: i: $i, j: $j, x[i] = $x[i], deltas.hidden: $deltaj")
	    end
        end
    end

    @assert assert_not_NaN(x)
    
    @assert assert_not_NaN(derivatives)

    for i = 1:size(derivatives.hidden_output, 1)
        for j = 1:size(derivatives.hidden_output, 2)
            derivatives.hidden_output[i,j] = deltas.output[j] * activation.hidden[i]
        end
    end

    for i = 1:size(derivatives.input_output, 1)
        for j = 1:size(derivatives.input_output, 2)
            derivatives.input_output[i,j] = deltas.output[j] * x[i]
        end
    end

    return derivatives
end


function gradient(sln::SLN_MLL, x::Sample)
    @assert num_dimensions(sln) == length(x)
    g = zeros(num_labels(sln), num_dimensions(sln))

    # TODO: calculate

    # this needs to return a gradient for each output label, right?
    @assert size(g) == (num_labels(sln), num_dimensions(sln))
    return g
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

