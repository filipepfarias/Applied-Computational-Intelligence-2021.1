using Base: Float64
using Revise
using AppCompIntel 
using CSV
using DataFrames
using PrettyTables
using Statistics, StatsBase
using Plots
using Latexify, LaTeXStrings
using YeoJohnsonTrans
using BoxCoxTrans
using MultivariateStats

println("Running HW1 ...\n");
println("Loading Concrete dataset...\n");
pgfplotsx()

concrete_df = CSV.File(eval(@__DIR__)*"/../data/Concrete_Data.csv") |> DataFrame;
#filter!(col -> !any(value -> value == 0.0, col), concrete_df);
transform!(concrete_df, "Concrete compressive strength (MPa)" => ByRow(strength -> get_category(strength)) => "Category");


# open(eval(@__DIR__)*"/../data/ConcreteUCI.csv", "w") do file
#     CSV.write(file, concrete_df)
# end

strength_categories = Array(["not standard", "standard", "high strength"]);
predictor_names = Array(["Cement", "Blast Furnace Slag", "Fly Ash", "Water", "Superplasticizer", "Coarse Aggregate", "Fine Aggregate", "Age (day)"]);

concrete_matrix = Matrix{Float64}(concrete_df[:, predictor_names]);
concrete_matrix = (concrete_matrix .- mean(concrete_matrix, dims=1)) ./ std(concrete_matrix, dims=1)
#concrete_matrix_transf = mapslices(YeoJohnsonTrans.transform, concrete_matrix, dims=1)

concrete_pca = MultivariateStats.fit(MultivariateStats.PCA, concrete_matrix', pratio=1)
concrete_projected_matrix = MultivariateStats.transform(concrete_pca, concrete_matrix')

# group results by category
not_standard = concrete_projected_matrix[:, concrete_df[:, "Category"] .== 1]
standard = concrete_projected_matrix[:, concrete_df[:, "Category"] .== 2]
high_strength = concrete_projected_matrix[:, concrete_df[:, "Category"] .== 3]

pca_scatter_plot = scatter(not_standard[1,:], not_standard[2,:], marker=:circle,linewidth=0, label="not standard")
scatter!(standard[1,:], standard[2,:], marker=:circle, linewidth=0, label="standard")
scatter!(high_strength[1,:], high_strength[2,:], marker=:circle, linewidth=0, label="high strength")
plot!(pca_scatter_plot, xlabel="First principal component", ylabel="Second principal component", legendtitle="Concrete category", legend=:outertop, legendnrows=1)

save_if_isfile(pca_scatter_plot, "pca_scatter_plot.pdf")
save_if_isfile(plot(principalvars(concrete_pca), xlabel="Principal components", ylabel="Variance", legend = false), "pca_variance.pdf");


num_predictors   = length(predictor_names);
num_categories   = length(strength_categories);
num_observations = nrow(concrete_df);

predictors_statistics_df = DataFrame()
predictors_statistics_notstandard_df = DataFrame()
predictors_statistics_standard_df = DataFrame()
predictors_statistics_highstrength_df = DataFrame()

for predictor_name in predictor_names
    predictor_array = Array(concrete_df[:, predictor_name])
    predictor_array_notstandard = Array(concrete_df[concrete_df[:, "Category"] .== 1, predictor_name])
    predictor_array_standard = Array(concrete_df[concrete_df[:, "Category"] .== 2, predictor_name])
    predictor_array_highstrength = Array(concrete_df[concrete_df[:, "Category"] .== 3, predictor_name])
    append!(predictors_statistics_df, DataFrame(mean = mean(predictor_array), std = std(predictor_array), gamma = skewness(predictor_array)))
    append!(predictors_statistics_notstandard_df, DataFrame(mean = mean(predictor_array_notstandard), std = std(predictor_array_notstandard), gamma = skewness(predictor_array_notstandard)))
    append!(predictors_statistics_standard_df, DataFrame(mean = mean(predictor_array_standard), std = std(predictor_array_standard), gamma = skewness(predictor_array_standard)))
    append!(predictors_statistics_highstrength_df, DataFrame(mean = mean(predictor_array_highstrength), std = std(predictor_array_highstrength), gamma = skewness(predictor_array_highstrength)))
end

print(predictors_statistics_df)
print(predictors_statistics_notstandard_df)
print(predictors_statistics_standard_df)
print(predictors_statistics_highstrength_df)

#predictors_corr_matrix = cor(concrete_matrix, concrete_matrix)

# pyplot()

# f = heatmap(predictor_names, predictor_names, predictors_corr_matrix,xtickfontrotation=20,framestyle=:box,clim=(-1,1),color=:balance,aspect_ratio=:equal);
# save_if_isfile(f,"predictors_corr_matrix.pdf");

# for category in 0:num_categories
#     f = plot_monovariate_histograms(concrete_df, predictor_names, category, false)
#     save_if_isfile(f,"monovariate_histograms_$category.pdf");
# end

# for category in 0:0
#     f = plot_bivariate_scatters(concrete_df, predictor_names, category)
#     save_if_isfile(f,"bivariate_scatters_$category.pdf");
# end