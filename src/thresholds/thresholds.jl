typealias Probability Float64

################################################
#
#       Takes Multi-dimensional array of probabilities
#       Output Binary predictions optimal for F1
#       Calculates F1 and Accuracy
#
################################################

function micro_f1_calculate(probabilities::Array{Float64, 2}, truth::Array{Float64,2})
    predictions = micro_f1_threshold(probabilities)
    tp, tn, fp, fn = confusion_matrix(predictions[:], truth[:])
    return f1(tp, fp, fn)
end

function macro_f1_calculate(probabilities::Array{Float64, 2}, truth::Array{Float64,2})
    predictions = macro_f1_threshold(probabilities)
    n,m =size(predictions)

    macro_sum = 0
    for j = 1:m
        tp, tn, fp, fn = confusion_matrix(predictions[:,j][:], truth[:,j][:])
        macro_sum += f1(tp, fp, fn)
    end
    return macro_sum/m
end

function per_example_f1_calculate(probabilities::Array{Float64, 2}, truth::Array{Float64,2})
    predictions = per_example_f1_threshold(probabilities)
    n,m =size(predictions)

    per_example_sum = 0
    for i = 1:n
        tp, tn, fp, fn = confusion_matrix(predictions[i,:][:], truth[i,:][:])
        per_example_sum += f1(tp, fp, fn)
    end
    return per_example_sum/n

end

function accuracy_calculate(probabilities::Array{Float64, 2}, truth::Array{Float64,2})
    predictions = accuracy_threshold(probabilities)
    tp, tn, fp, fn = confusion_matrix(predictions[:], truth[:])
    return (tp + tn ) / (tp + tn + fp + fn)

end


function confusion_matrix(predictions::Array{Float64}, truth::Array{Float64})
    @assert length(predictions) == length(truth)
    n = length(predictions)
    tp=tn=fp=fn=0

    for i = 1:n
        if predictions[i] == 1.0
            if truth[i] == 1.0
                tp += 1
            else
                fp += 1
            end
        else
            if truth[i] == 1.0
                fn += 1
            else
                tn += 1
            end
        end
    end

    return tp,tn,fp,fn
end

function f1(tp::Number, fp::Number, fn::Number)
    return 2tp / (2tp + fp + fn)
end


function accuracy_threshold(probabilities::Array{Probability, 2})
    n, m = size(probabilities)
    predictions = zeros(n,m)

    for i = 1:n
        for j = 1:m
            predictions[i,j] = if (probabilities[i, j] > .5) 1.0 else 0 end
        end
    end
    return predictions
end


function micro_f1_threshold(probabilities::Array{Probability, 2})

    linprob = probabilities[:]
    threshold = f1_threshold(linprob)
    predictions = apply_threshold(probabilities, threshold)
    return predictions

end


function macro_f1_threshold(probabilities::Array{Probability, 2})
    n, m = size(probabilities)
    predictions = zeros(n,m)
    for j = 1:m
        label_probabilities = probabilities[:,j][:]
        threshold = f1_threshold(label_probabilities)
        predictions[:,j]  = apply_threshold(label_probabilities, threshold)
    end
    return predictions
end

function per_example_f1_threshold(probabilities::Array{Probability, 2})
    n,m = size(probabilities)
    predictions = zeros(n,m)
    for i = 1:n
        example_probabilities = probabilities[i,:][:]
        threshold = f1_threshold(example_probabilities)
        predictions[i,:]  = apply_threshold(example_probabilities, threshold)
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
