

######################################################
#  Log loss for a single example
######################################################
epsilon = 1e-9

function log_loss(y::Float64, y_hat::Float64)
    if y == y_hat
        return 0.0
    else
        if y_hat <= 0.0
           y_hat = epsilon
        elseif y_hat >= 1.0
           y_hat = 1.0 - epsilon
        end
        return -(y * log(y_hat) + (1 - y) * log(1 - y_hat))
    end
end

######################################################
#  Log loss for a set of examples
######################################################
function log_loss(y::Array{Float64}, y_hat::Array{Float64})
    assert(length(y) == length(y_hat))
    n = length(y)
    sum = 0
    for i = 1:n
        sum += log_loss(y[i], y_hat[i])
    end
    return 1/n * sum
end


function log_loss_prime(y::Float64, y_hat::Float64)
    if y == y_hat
	return 0.0
    else
	ll = (y/y_hat + (y-1)/(1-y_hat))
	if ll > 10000
	    ll = 10000
	elseif ll < -10000
	    ll = -10000
	end
    end
    return ll
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
    if sum == 0
        return 0
    else
        return 1
    end
end










