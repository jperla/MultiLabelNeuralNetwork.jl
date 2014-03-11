using Base.Test

import NeuralNetworks: SLN_MLL, forward_propagate, gradient
import CheckGrad: checkgrad, approximately_one


sln = SLN_MLL(10, 3, 2)
x0 = [1, 0, 0, 1, 0.5, 1.0, 0.0, 1.0, 0.0, 1.0]
q(x) = (forward_propagate(sln, x)[1], gradient(sln, x)[1,:])

@test approximately_one(checkgrad(q, x0))


@test log_loss(1.0,0.0) == Inf

@test approximately_one(log_loss(1.0,.99999999))
