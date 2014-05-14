import ArgParse: ArgParseSettings, @add_arg_table, parse_args
import NeuralNetworks: SLN_MLL, SLN_MLL_Activation, SLN_MLL_Deltas, SLN_MLL_Derivatives,
                       read_data, flat_weights!, flat_weights_length,
                       log_loss, assert_not_NaN,
                       whiten, prepend_intercept,
                       TanhLinkFunction, RectifiedLinearUnitLinkFunction, SigmoidLinkFunction

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "dataset"
            help = "The dataset we want to use: emotions, scene, yeast, or reuters."
            arg_type = String
            required = true
        "--hidden", "-H"
            help = "The number of hidden nodes."
            arg_type = Integer
            default = 10
        "--eta0"
            help = "The initial learning rate."
            arg_type = FloatingPoint
            default = 0.5
        "--adagrad"
            help = "Use adagrad dynamic learning rate."
            action = :store_true
        "--sparse"
            help = "Sparsify input data"
            action = :store_true
        "--time"
            help = "Output timings for each iteration."
            action = :store_true
        "--epochs", "-e"
            help = "Number of epochs to do."
            arg_type = Integer
            default = 100
        "--rio"
            help = "Regularization constant between input and output layers."
            arg_type = FloatingPoint
            default = 0.0
        "--rih"
            help = "Regularization constant between input and hidden layers."
            arg_type = FloatingPoint
            default = 0.0
        "--rho"
            help = "Regularization constant between hidden and output layers."
            arg_type = FloatingPoint
            default = 0.0
        "--interval", "-i"
            help = "How frequently to print progress."
            arg_type = Integer
            default = 10000
        "--file", "-f"
	    help = "Save weights at each interval to a file instead of calculating losses and printing to screen."
	    arg_type = String
	    default = ""
        "--tanh"
	    help = "Use TanH linked function (recommended by LeCunn) instead of Rectified Linear Units"
            action = :store_true
        "--dropout"
            help = "Randomly zero out hidden nodes during training."
            action = :store_true
    end

    return parse_args(s)
end

function slnmll_from_args(dimensions::Int, nlabels::Int, parsed_args::Dict)
    RUNT = Float64

    hidden_nodes = parsed_args["hidden"]
    initial_learning_rate = parsed_args["eta0"]
    adagrad = parsed_args["adagrad"]
    rio = parsed_args["rio"]
    rih = parsed_args["rih"]
    rho = parsed_args["rho"]
    dropout = parsed_args["dropout"]
    tanh_link = parsed_args["tanh"]

    hidden_link = if tanh_link TanhLinkFunction() else RectifiedLinearUnitLinkFunction() end
    sln = SLN_MLL(RUNT, dimensions, nlabels, hidden_nodes, hidden_link, SigmoidLinkFunction())

    if adagrad
        @printf("SLN MLL AdaGrad\n")
        slnmll = MultilabelSLNAdaGrad(sln, initial_learning_rate=initial_learning_rate, rio=rio, rih=rih, rho=rho, dropout=dropout)
    else
        @printf("SLN MLL SGD\n")
        slnmll = MultilabelSLNSGD(sln, initial_learning_rate=initial_learning_rate, rio=rio, rih=rih, rho=rho, dropout=dropout)
    end
    return slnmll
end
