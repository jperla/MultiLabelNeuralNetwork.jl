import NeuralNetworks: log_loss
import Thresholds: accuracy_calculate, micro_f1_calculate, zero_one_calculate

function read_weights(filename::String)
    data = readdlm(filename, ' ', Float64)
    samples = Int[int64(data[i,1]) for i in 1:size(data, 1)]
    weights = data[1:end, 2:end]
    return (samples, weights)
end

function calculate_losses{T, U<:FloatingPoint, W<:FloatingPoint}(classifier::GradientScratch{T}, weights::AbstractMatrix{T},
                                                                 X::AbstractMatrix{U}, Y::AbstractMatrix{W})
    niter = size(weights, 1)
    y_hat = zeros(T, size(Y))
    losses = zeros(T, niter, 4)

    for i in 1:niter
        calculate_losses!(classifier, weights, X, Y, y_hat, losses, i)
    end

    return losses
end

function calculate_losses!{T, U<:FloatingPoint, W<:FloatingPoint}(classifier::GradientScratch{T}, weights::AbstractMatrix{T},
                                                                  X::AbstractMatrix{U}, Y::AbstractMatrix{W}, 
                                                                  y_hat::AbstractMatrix{W}, 
                                                                  losses::AbstractMatrix{Float64}, i::Int)
    predict!(classifier, sub(weights, (i, 1:size(weights, 2))), X, y_hat)

    ll = log_loss(y_hat, Y)
    accuracy = accuracy_calculate(y_hat, Y)
    micro_f1 = micro_f1_calculate(y_hat, Y)
    zero_one = zero_one_calculate(y_hat, Y)

    losses[i, 1] = ll
    losses[i, 2] = accuracy
    losses[i, 3] = micro_f1
    losses[i, 4] = zero_one
end

