import MultilabelNeuralNetwork: MultilabelSLN, MultilabelSLNAdaGrad, MultilabelSLNSGD, predict, calculate_gradient!
import NeuralNetworks: read_data, flat_weights
import StochasticGradient: train_samples

function dataset_log_loss(g, w, X, Y)
    y_hat = zeros(Float64, (size(Y, 1), num_labels(g)))
    for i in 1:size(Y, 1)
        y_hat[i,:] = predict(g, w, X[i,:][:])
    end
    loss = log_loss(Y, y_hat)
    return loss
end

function learn(g, X, Y, w, num_iter=100)                                                                                                
    loss = 1.0                                                                                                                          
    for j in 1:num_iter                                                                                                                 
        loss = dataset_log_loss(g, w, X, Y)                                                                                             
        if (j % int(num_iter / 10) == 1)                                                                                                
            @printf("Epoch %i (loss %4f)\n", j, loss)
        end

        for i in 1:size(Y, 1)
            y = Y[i,:][:]
                train_samples!(g, w, X[i,:], y, j)
            end
        end
    end
end

features, labels = read_data("emotions", "train")

dimensions = size(features, 2)
nlabels = length(unique(labels))
@printf("SLN MLL SGD\n")
sln = SLN_MLL(dimensions, nlabels, 2)
mweights = flat_weights(sln)
slnmllsgd = MultilabelSLNSGD{Float64}(zeros(Float64, length(mweights)), nlabels, 1.0, sln)

@time learn(slnmllsgd, features, labels, mweights, 100)

