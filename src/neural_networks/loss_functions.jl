

######################################################
#  Log loss for a single example
######################################################
function log_loss(y::Float64, y_hat::Float64)
    return -(y * log(y_hat) + (1 - y) * log(1 - y_hat))
end

######################################################
#  Log loss for a set of examples
######################################################
function log_loss(y::Array{Float64}, y_hat::Array{Float64})
    assert(length(y) == length(y_hat))
    n = length(y)
    sum = 0
    for i = 1:n
        sum += log_loss(y, y_hat)
    end
    return 1/n * sum
end


function log_loss_prime(y::Float64, y_hat::Float64)
    return -1 * (y/y_hat + (y-1)/(1-y_hat))
end


function square_loss(y::Float64, y_hat::Float64)
    return (y-y_hat) ^ 2
end

function square_loss(y::Array{Float64}, y_hat::Array{Float64})
    sum = 0
    n = length(y)
    for i = 1:length(y)
        sum += square_loss(y[i], y_hat[i])
    end
    return 1/n * sum
end

function square_loss_prime(y::Float64, y_hat::Float64)
    return -2y + 2y_hat
end

function zero_one_loss(y::Float64, y_hat::Float64)
    return int( y != y_hat)
end

function zero_one_loss(y::Array{Float64}, y_hat::Array{Float64})
    sum = 0
    n = length(y)
    for i = 1:n
        sum += zero_one_loss(y[i], y_hat[i])
    end
    return 1/n * sum
end


