

################################################
# TODO: Fill in these thresholding functions
#       Take Multi-dimensional array of probabilities
#       Output Binary predictions optimal for F1
################################################

function macro_f1_threshold(probabilities::Array{Probability, 2})

    linprob = probabilities[:]
    linprob = sort(linprob)

    return true
end


function micro_f1_threshold(probabilities::Array{Probability, 2})

end

function per_example(probabilities::Array{Probability, 2})

end

function f1(probabilities::Array{Probability})

end
