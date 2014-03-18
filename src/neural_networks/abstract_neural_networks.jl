# Useful for efficient indexing
typealias VectorOrSubArrayVector{T} Union(Vector{T}, SubArray{T})

typealias Weight Float64
typealias Weights Matrix{Weight}

typealias Activation Float64
typealias Activations Vector{Activation}

typealias Sample Vector{Float64}

typealias Probability Float64
typealias Labels Vector{Probability} # multi-label setting

abstract NeuralNetworkStorage
