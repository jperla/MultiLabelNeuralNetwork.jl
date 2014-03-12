using Base.Test
import Calculus: check_gradient

import CheckGrad: checkgrad, approximately_one

epsilon = 1e-6

f(x) = (sum(x .^ 2), 2 .* x)

x1 = [3.0, 4.0, 5.0]
@test check_gradient(x->f(x)[1], x->f(x)[2], x1) < epsilon 
@test approximately_one(checkgrad(f, x1))

g(x) = (sum(x .^ 2), 3 .* x)
@test check_gradient(x->g(x)[1], x->g(x)[2], x1) > epsilon
@test !approximately_one(checkgrad(g, x1))

