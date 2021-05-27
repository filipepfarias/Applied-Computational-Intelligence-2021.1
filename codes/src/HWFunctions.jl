module HWFunctions

    function plot_monovariate_histograms(df, predictor_names, category = 0)
        num_predictors = length(predictor_names)
        plots = Array{Any}(undef, num_predictors)
        if category == 0
            for i in 1:num_predictors
                plots[i] = plot(df[:,i], t = [:histogram, :density], title=predictor_names[i], normed=true)
            end
        else
            classconditioned_df = subset(df, :Category => ByRow(==(category)))
            for i in 1:num_predictors
                plots[i] = plot(classconditioned_df[:,i], t = [:histogram, :density], title=predictor_names[i], normed=true)
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
                    plot_matrix[i,j] = plot(df[:,i], t = [:histogram, :density], normed=true)
                else
                    plot_matrix[i,j] = scatter(df[:,i], df[:,j])
                end
            end    
        end
        return plot_matrix
    end

    function plot_bivariate_scatters(df, predictor_names, category = 0)
        if category == 0
            plot_matrix = plot_scatters(df, predictor_names)
            return plot(plot_matrix[:]..., layout=(num_predictors,num_predictors), size=(1500,1500), axis=false, ticks=false, legend=false)
        else
            classconditioned_df = subset(df, :Category => ByRow(==(category)))
            plot_matrix = plot_scatters(classconditioned_df, predictor_names)
            return plot(plot_matrix[:]..., layout=(num_predictors, num_predictors),size=(1500,1500), axis=false, ticks=false, legend=false)
        end
    end

    export plot_monovariate_histograms
    export plot_bivariate_scatters

end # module
