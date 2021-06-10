function get_category(strength)
    if strength < 25 # Non-standard -> https://www.baseconcrete.co.uk/different-types-of-concrete-grades-and-their-uses/
        return 1
    elseif 25 <= strength < 50 # Standard
        return 2
    elseif  50 <= strength # High Strength
        return 3
    end
end

function boxcox_transform(x,λ)
    if λ == 0
        return log(x)
    else
        return (x^λ - 1)/λ
    end
end

function yeojohnson_transform(x,λ,sign_data = 1)
    if λ == 0
        return sign_data*log(sign_data*x+1)
    else
        return (sign_data*(x+1)^λ - 1)/λ
    end 
end

function figure_path(figure_name)
    return eval(@__DIR__)*"/../../hw1/figures/"*figure_name
end

function save_if_isfile(f,path)
    if ~isfile(figure_path(path))
        savefig(f,figure_path(path))
    end    
end

function plot_monovariate_histograms(df, predictor_names, category = 0, trans = false)
    num_predictors = length(predictor_names)
    plots = Array{Any}(undef, num_predictors)
    if category == 0
        for i in 1:num_predictors
            if trans
                plots[i] = plot(YeoJohnsonTrans.transform(df[:,i]), t = [:histogram], title=predictor_names[i], normed=true)
            else
                plots[i] = plot(df[:,i], t = [:histogram], title=predictor_names[i], normed=true)
            end
        end
    else
        class_df = subset(df, :Category => ByRow(==(category)))
        for i in 1:num_predictors
            if trans
                plots[i] = plot(YeoJohnsonTrans.transform(class_df[:,i]), t = [:histogram], title=predictor_names[i], normed=true)
            else
                plots[i] = plot(class_df[:,i], t = [:histogram], title=predictor_names[i], normed=true)
            end
        end
    end
    return plot(plots[:]..., layout = grid(2, Int(num_predictors / 2)), size = (1000, 1000), legend=false)
end

function plot_scatters(df, predictor_names)
    num_predictors = length(predictor_names)
    plot_matrix = Matrix{}(undef,num_predictors,num_predictors);
    for i in 1:num_predictors
        for j in 1:num_predictors
            if i == j
                # plot_matrix[i,j] = plot(df[:,i], t = [:histogram, :density], normed=true)
                plot_matrix[i,j] = plot(df[:,i], t = [:histogram], normed=true)
            else
                plot_matrix[i,j] = scatter(df[:,i], df[:,j])
            end
        end    
    end
    return plot_matrix
end

function plot_bivariate_scatters(df, predictor_names, category = 0)
    num_predictors = length(predictor_names)
    if category == 0
        plot_matrix = plot_scatters(df, predictor_names)
        return plot(plot_matrix[:]..., layout=(num_predictors,num_predictors), size=(1500,1500), axis=false, ticks=false, legend=false)
    else
        classconditioned_df = subset(df, :Category => ByRow(==(category)))
        plot_matrix = plot_scatters(classconditioned_df, predictor_names)
        return plot(plot_matrix[:]..., layout=(num_predictors, num_predictors),size=(1500,1500), axis=false, ticks=false, legend=false)
    end
end