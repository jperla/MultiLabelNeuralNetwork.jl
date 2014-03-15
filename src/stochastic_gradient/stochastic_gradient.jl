
typealias AdaGrad Vector{Float64}

function train_binary_sample!(weights::Vector{Float64}, 
                              gradient_func!::Function, scratch_gradient::Vector{Float64}, 
                              x::Vector{Float64}, y::Float64,
                              eta0::Float64, # initial learning rate
                              t::Int # iteration number
                             )
    # Accepts one stochastic sample from the distribution, and updates the weight vector given gradient function.
    # Uses the scratch_gradient as a workspace to avoid memory allocation.
    # We assume that x already has an intercept term (constant 1) as the first value.
    @assert length(scratch_gradient) == length(x)
    gradient_func!(weights, x, y, scratch_gradient) # fill scratch gradient
    weights += ((eta0 / sqrt(t)) .* scratch_gradient)
end

function train_sample!(weights,
                       gradient_func!::Function, scratch_gradient
                       x::Vector{Float64}, y,
                       eta0::Float64, # initial learning rate
                       t::Int # iteration number
                      )
    # Accepts one stochastic sample from the distribution, and updates the weight vector given gradient function.
    # Uses the scratch_gradient as a workspace to avoid memory allocation.
    # We assume that x already has an intercept term (constant 1) as the first value.
    gradient_func!(weights, x, y, scratch_gradient) # fill scratch gradient
    weights += ((eta0 / sqrt(t)) .* scratch_gradient)
end

function train_sample_adagrad!(weights,
                               gradient_func!::Function, scratch_gradient, scratch_adagrad::AdaGrad,
                               x::Vector{Float64}, y,
                               eta0::Float64, # initial learning rate
                               t::Int # iteration number
                              )
    # The same as train_sample! but it uses AdaGrad to have a dynamic learning rate
    gradient_func!(weights, x, y, scratch_gradient) # fill scratch gradient
    update_scratch_adagrad(scratch_adagrad, scratch_gradient) # keep running sum of diagonal
    weights += (eta0 .* learning_rate(scratch_adagrad) .* scratch_gradient)
end

function update_scratch_adagrad(weights::AdaGrad, gradient::Vector{Float64})
    weights += sum(gradient * gradient', 1)
end

function learning_rate(weights::AdaGrad)
    1.0 / sqrt(weights)
end
