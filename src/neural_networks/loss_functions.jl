

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
    for i = 1:length(y)
        sum += log_loss(y, y_hat)
    end
    return 1/n * sum
end

