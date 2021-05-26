using DataFrames, CSV, Plots, StatsBase, Statistics

concrete_df = dropmissing(CSV.File("./data/ConcreteUCI.csv") |> DataFrame)

strength_categories = Array(["very low", "low", "medium", "high", "very high"])
predictor_names = Array(["Cement", "Blast Furnace Slag", "Fly Ash", "Water", "Superplasticizer", "Coarse Aggregate", "Fine Aggregate", "Age (day)"])
num_predictors = length(predictor_names)

concrete_matrix = Matrix{Real}(concrete_df[:,predictor_names])
predictors_corr_matrix = cor(concrete_matrix, concrete_matrix)

f1 = Plots.heatmap(predictor_names, predictor_names, predictors_corr_matrix, xrotation = 45)

plot_matrix = Matrix{}(undef,num_predictors,num_predictors);

for i in 1:num_predictors
    for j in 1:num_predictors
        if i == j
            plot_matrix[i,j] = histogram(concrete_df[:,i])
        else
            plot_matrix[i,j] = scatter(concrete_df[:,i],concrete_df[:,j])
        end
    end    
end

f2 = plot(plot_matrix[:]..., layout=(num_predictors,num_predictors), size=(3000,3000), axis=false, ticks=false, legend=false)

savefig(p2,"./hw1/figures/matrix-corplot.pdf");