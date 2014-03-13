

################################################
# TODO: Fill in these thresholding functions
#       Take Multi-dimensional array of probabilities
#       Output Binary predictions optimal for F1
################################################

function micro_f1(probabilities::Array{Probability, 2})

    linprob = probabilities[:]
    threshold = f1_threshold(linprob)
    predictions = apply_threshold(probabilities, threshold)
    return predictions

end


function macro_f1(probabilities::Array{Probability, 2})
    n,m = size(probabilities)
    predictions = zeros(n,m)
    for j = 1:m
        label_probabilities = probabilities[1,j][:]
        threshold = f1_threshold(label_probabilities)
        predictions[:,j]  = apply_threshold(label_probabilities, threshold)
    end
    return predictions
end

function per_example(probabilities::Array{Probability, 2})
    n,m = size(probabilities)
    predictions = zeros(n,m)
    for i = 1:n
        example_probabilities = probabilities[i,:][:]
        threshold = f1_threshold(label_probabilities)
        predictions[i,:]  = apply_threshold(label_probabilities, threshold)
    end
    return predictions
end


function apply_threshold(probabilities::Array{Probability}, threshold::Float64)

    n = length(probabilities)
    predictions = zeros(n)
    for i = 1:n
        predictions[i] = if probabilities[i] > threshold 1 else 0 end
    end
    return predictions
end

function apply_threshold(probabilities::Array{Probability, 2}, threshold::Float64)
    n, m = size(probabilities)
    predictions = zeros(n,m)
    for i = 1:n
        for j=1:m
            predictions[i,j] = if probabilities[i,j] > threshold 1 else 0 end
        end
    end
    return predictions
end

function f1_threshold(probabilities::Array{Probability})
    sorted_probabilities = sort(probabilities)
    max_f1 = 0
    numerator = 0
    denominator = sum(sorted_probabilities)
    n = length(sorted_probabilities)
    expected_f1 =0

    for i = 1:length(sorted_probabilities)
        numerator += 2(sorted_probabilities[i])
        denominator += 1
        expected_f1 = numerator / denominator
        if expected_f1 > max_f1
            max_f1 = expected_f1
        end
    end
    return max_f1/2

end
