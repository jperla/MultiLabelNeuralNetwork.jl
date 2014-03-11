typealias Weights Matrix{Float64}
typealias Sample Vector{Float64}

type SLN_MLLStorage
    # single layer neural network for multi label learning with skip level weights
    input_output::Weights
    input_hidden::Weights
    hidden_output::Weights
end

function SLN_MLLStorage(num_dimensions::Int, num_labels::Int, num_hidden::Int)
    input_output = randn(num_dimensions, num_labels)
    input_hidden = randn(num_dimensions, num_hidden)
    hidden_output = randn(num_hidden, num_labels)
    SLN_MLLStorage(input_output, input_hidden, hidden_output)
end

# Size helper functions
num_dimensions(sln::SLN_MLLStorage) = size(sln.input_output, 1)
num_hidden(sln::SLN_MLLStorage) = size(sln.input_hidden, 2)
num_output(sln::SLN_MLLStorage) = size(sln.input_output, 2)
num_labels(sln::SLN_MLLStorage) = num_output(sln)

# Link functions
relu(x) = max(0, x)
@vectorize_1arg Number relu

sigmoid(x) = 1.0 / (1.0 + e^(-x))
@vectorize_1arg Number sigmoid

function forward_propagate(sln::SLN_MLLStorage, x::Sample)
    @assert length(x) == num_dimensions(sln)
    hidden = relu(x' * sln.input_hidden)
    skip_activation = x' * sln.input_output
    hidden_activation = hidden * sln.hidden_output
    label_probabilities = sigmoid(hidden_activation .+ skip_activation)
    @printf("lp: %i %s", length(label_probabilities), label_probabilities)
    @assert length(label_probabilities) == num_labels(sln)
    return label_probabilities
end

function checkgrad(f::Function, x::Sample)
     e = 1e-6
     n = length(x)
     y, dx = f(x)
     d = e .* sign((2 .* rand(n)) - 1)

     y2, dx2 = f(x .+ d)
     y1, dx1 = f(x .- d)

     r = (y2 - y1) / (2 .* sum(d .* dx))

     @printf("e=%f ratio=%f (should be extremely close to unity)\n", e, r)
end

f(x) = (sum(x .^ 2), 2 .* x)

x1 = [3.0, 4.0, 5.0]
checkgrad(f, x1)
# this should fail:
g(x) = (sum(x .^ 2), 3 .* x)
checkgrad(g, x1)

function gradient(sln::SLN_MLLStorage, x::Sample)
   # TODO
end

sln = SLN_MLLStorage(10, 3, 2)
x0 = [1, 0, 0, 1, 0.5, 1.0, 0.0, 1.0, 0.0, 1.0]
q(x) = (forward_propagate(sln, x), gradient(sln, x))
checkgrad(q, x0)

