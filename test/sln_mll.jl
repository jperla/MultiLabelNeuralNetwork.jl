using Base.Test

import NeuralNetworks: SLN_MLL, forward_propagate, gradient, zero!
import CheckGrad: checkgrad, approximately_one

sln = SLN_MLL(10, 3, 2)
@assert sln.input_output[1] != 0.0
zero!(sln)
@assert sln.input_output[1] == 0.0
