using Base.Test

import Thresholds: f1_threshold, micro_f1, macro_f1, per_example_f1
import CheckGrad: checkgrad, approximately_one
import NeuralNetworks



@test f1_threshold(ones(10)) == .5
@test approximately_one(f1_threshold([.5, .5, .5, .5]) / .333333333333333333)


