abstract LinkFunction

typealias WeightMatrix{T} Matrix{T}
typealias WeightVector{T} Vector{T}

typealias Activations{T} Vector{T}

typealias Sample Vector{Float64}

typealias Probability Float64
typealias Labels Vector{Probability} # multi-label setting

abstract NeuralNetworkStorage{T}
