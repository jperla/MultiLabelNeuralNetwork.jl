
######################################################
#  Log loss for a set of examples
######################################################

function log_loss(y::Array{Float64}, y_hat::Array{Float64})

    return true

end

######################################################
#  Log loss for a single example
######################################################
function log_loss(y::Float64, y_hat::Float64)

    return -(y * log(y_hat) + (1 - y) * log(1 - y_hat))

end


function soft_plus(activation)



end
