using DataFrames, CSV, Plots, StatsBase, Statistics, Formatting, LaTeXStrings

concrete_df = dropmissing(CSV.File("../../data/ConcreteUCI.csv") |> DataFrame)
strength_categories = Array(["very low", "low", "medium", "high", "very high"])
predictor_names = Array(["Cement", "Blast Furnace Slag", "Fly Ash", "Water", "Superplasticizer", "Coarse Aggregate", "Fine Aggregate", "Age (day)"])
concrete_matrix = Matrix{Real}(concrete_df[:,predictor_names]);

num_predictors = length(predictor_names)
num_categories = length(strength_categories)
num_observations = nrow(concrete_df)

latexify("D = %$num_predictors") |> display
latexify("L = %$num_categories") |> display

for i in 1:num_categories
    n = nrow(subset(concrete_df, :Category => ByRow(==(i))))
    latexify(L"N_{L_{%$i}} = %$n") |> display
end

predictors_statistics_df = DataFrame()

for predictor_name in predictor_names
    predictor_array = Array(concrete_df[:, predictor_name])
    append!(predictors_statistics_df, DataFrame(mean = mean(predictor_array), std = std(predictor_array), gamma = skewness(predictor_array)))
end
print(predictors_statistics_df)

predictors_corr_matrix = cor(concrete_matrix, concrete_matrix)
pyplot()
f1 = heatmap(predictor_names, predictor_names, predictors_corr_matrix,xtickfontrotation=20,framestyle=:box,clim=(-1,1),color=:balance,aspect_ratio=:equal,size=(800,720))
savefig(f1,"../figures/predictors_corr_matrix.pdf");

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

savefig(f2,"../figures/matrix-corplot.pdf");