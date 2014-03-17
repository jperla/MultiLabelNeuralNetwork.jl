import StochasticGradient: train_samples!
import NeuralNetworks: read_data, flat_weights, SLN_MLL, log_loss
import MultilabelNeuralNetwork: MultilabelSLN, MultilabelSLNAdaGrad, MultilabelSLNSGD, predict, calculate_gradient!

function num_labels(g)
    g.num_labels
end

function dataset_log_loss(g, w, X, Y)
    y_hat = zeros(Float64, (size(Y, 1), num_labels(g)))
    for i in 1:size(Y, 1)
        y_hat[i,:] = predict(g, w, X[i,:][:])
    end
    loss = log_loss(Y, y_hat)
    return loss
end

function learn(g, w, X, Y; epochs=100, modn=10)
    loss = 1.0
    for e in 1:epochs
        loss = dataset_log_loss(g, w, X, Y)
        if ((modn == 1) || (e % modn == 1))
            @printf("Epoch %i (loss %4f): %s\n", e, loss, w[1:20]')
        end

        for i in 1:size(Y, 1)
            y = Y[i,:][:]
            train_samples!(g, w, X[i,:], y, e)
        end
    end
end

emotions_train_features, emotions_train_labels = read_data("emotions", "train")

dimensions = size(emotions_train_features, 2)
nlabels = size(emotions_train_labels, 2)
@assert size(emotions_train_labels, 1) == size(emotions_train_features, 1)
hidden_nodes = 3
@printf("SLN MLL SGD\n")
sln = SLN_MLL(dimensions, nlabels, hidden_nodes)
mweights = flat_weights(sln)
slnmllsgd = MultilabelSLNSGD{Float64}(zeros(Float64, length(mweights)), nlabels, 0.01, sln)

@time learn(slnmllsgd, mweights, emotions_train_features, emotions_train_labels, epochs=3, modn=1)


@printf("SLN MLL AdaGrad\n")
sln = SLN_MLL(dimensions, nlabels, hidden_nodes)
mweights = flat_weights(sln)
slnmllada = MultilabelSLNSGD{Float64}(zeros(Float64, length(mweights)), nlabels, 0.01, sln)

@time learn(slnmllada, mweights, emotions_train_features, emotions_train_labels, epochs=3, modn=1)

