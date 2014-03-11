using Base.Test

import CheckGrad: checkgrad, approximately_one

f(x) = (sum(x .^ 2), 2 .* x)

x1 = [3.0, 4.0, 5.0]
@test approximately_one(checkgrad(f, x1))

g(x) = (sum(x .^ 2), 3 .* x)
@test !approximately_one(checkgrad(g, x1))

