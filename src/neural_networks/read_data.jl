
function read_data(x::String, mode::String)
    num_features = num_labels = 0

    if x == "scene"
        num_features = 294
        num_labels = 6
    elseif x == "yeast"
        num_features = 103
        num_labels = 14
    elseif x == "emotions"
        num_features = 72
        num_labels = 6
    elseif x == "reuters"
	num_features = 18637
	num_labels = 90
    else
        error("Unknown dataset")
    end

    if mode == "train"
        suffix = "train"
    else
        suffix = "test"
    end

    path = joinpath("data", x, string(x, "-", suffix, ".csv"))
    file = readdlm(path, ',', Float64)

    n = size(file, 1)
    features = file[1:n, 1:num_features]
    labels = file[1:n, (num_features + 1):(num_features + num_labels)]

    return (features, labels)
end

