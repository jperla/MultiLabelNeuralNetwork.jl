using Base.Test

import Thresholds: f1_threshold

@test f1_threshold(ones(10)) == .5
@test f1_threshold([.5, .5, .5, .5]) == 0.33333333333333333


