
function read_data(x::String, mode::String)

    num_features = num_labels = 0

    if x == "scene"
        num_features = 294
        num_labels = 6
    end

    if x == "yeast"
        num_features = 103
        num_labels = 14
    end

    if x == "emotions"
        num_features = 72
        num_labels = 6

    end

    if mode == "train"
        suffix = "-train.csv"
    else
        suffix = "-test.csv"
    end

    path = string("data/", x, "/", x, suffix)
    file = readdlm(path, ',')

    n = size(file, 1)
    features = file[1:n, 1:num_features]
    labels = fil[1:n, (num_features + 1):(num_features+num_labels))

    return (features, labels)

end

