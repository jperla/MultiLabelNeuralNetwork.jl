using Base.Test

import Thresholds: f1_threshold, zero_one_calculate

@test f1_threshold(ones(10)) == .5
@test f1_threshold([.5, .5, .5, .5]) == 0.33333333333333333

x = [1.0 .77 .65; .55 .78 .99]
y = [1.0 0.0 1.0; 1.0 0.0 1.0]
@test zero_one_calculate(x,y) == 1.0
y[2,2] == 1.0
@test zero_one_calculate(x,y) == 0.5



