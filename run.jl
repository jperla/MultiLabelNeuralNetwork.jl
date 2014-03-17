import StochasticGradient: train_samples!
import NeuralNetworks: read_data, flat_weights, SLN_MLL, log_loss
import MultilabelNeuralNetwork: MultilabelSLN, MultilabelSLNAdaGrad, MultilabelSLNSGD, predict, calculate_gradient!
import Thresholds: accuracy_calculate, micro_f1_calculate

function num_labels(g)
    g.num_labels
end

function dataset_log_loss(g, w, X, Y)
    y_hat = zeros(Float64, (size(Y, 1), num_labels(g)))
    for i in 1:size(Y, 1)
        y_hat[i,:] = predict(g, w, X[i,:][:])
    end
    loss = log_loss(Y, y_hat)
    if size(Y, 2) > 1
        micro_f1 = micro_f1_calculate(y_hat, Y)
        accuracy = accuracy_calculate(y_hat, Y)
    else
        micro_f1 = accuracy = 0
    end                                                                                                                                
    return loss, micro_f1, accuracy    
end

function learn(g, w, X, Y; epochs=100, modn=10)
    loss = 1.0
    for e in 1:epochs
        loss, micro_f1, accuracy = dataset_log_loss(g, w, X, Y)
        if ((modn == 1) || (e % modn == 1))
            @printf("Epoch %i (loss %4f): %s", e, loss, w[1:3]')
            @printf("\t Micro_F1: %4f,  Accuracy: %4f \n", micro_f1, accuracy)
        end

        for i in 1:size(Y, 1)
            y = Y[i,:][:]
            train_samples!(g, w, X[i,:], y, e)
        end
    end
end

function whiten{T<:Number}(m::Array{T, 2})
    m = broadcast(-, m, mean(m, 1))
    m = broadcast(/, m, std(m, 1))
    return m
end

function prepend_intercept{T<:Number}(m::Array{T, 2})
    return hcat(ones(T, size(m, 1)), m)
end

emotions_train_features, emotions_train_labels = read_data("emotions", "train")
emotions_train_features = whiten(emotions_train_features)
emotions_train_features = prepend_intercept(emotions_train_features)

dimensions = size(emotions_train_features, 2)
nlabels = size(emotions_train_labels, 2)
@assert size(emotions_train_labels, 1) == size(emotions_train_features, 1)
hidden_nodes = 3
@printf("SLN MLL SGD\n")
sln = SLN_MLL(dimensions, nlabels, hidden_nodes)
mweights = flat_weights(sln)
slnmllsgd = MultilabelSLNSGD{Float64}(zeros(Float64, length(mweights)), nlabels, 0.01, sln)

nepochs = 1000

@time learn(slnmllsgd, mweights, emotions_train_features, emotions_train_labels, epochs=nepochs, modn=1)

@printf("SLN MLL AdaGrad\n")
sln = SLN_MLL(dimensions, nlabels, hidden_nodes)
mweights = flat_weights(sln)
slnmllada = MultilabelSLNAdaGrad{Float64}(zeros(Float64, length(mweights)), nlabels, 0.1, sln, ones(Float64, length(mweights)))

@time learn(slnmllada, mweights, emotions_train_features, emotions_train_labels, epochs=nepochs, modn=1)

