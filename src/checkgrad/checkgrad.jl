function checkgrad(f::Function, x::Vector{Float64})
    e = 1e-6
    n = length(x)
    y, dx = f(x)
    d = e .* sign((2 .* rand(n)) - 1)

    y2, dx2 = f(x .+ d)
    y1, dx1 = f(x .- d)

    r = (y2 - y1) / (2 .* sum(d .* dx))

    #@printf("e=%f ratio=%f (should be extremely close to unity)\n", e, r)
    return r
end

approximately_one(x) = abs(x - 1.0) < 1e-6

