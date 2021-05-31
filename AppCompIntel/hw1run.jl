using Revise
using AppCompIntel 
using CSV
using DataFrames
using PrettyTables
using Statistics, StatsBase
using Plots
using Latexify, LaTeXStrings

println("Running HW1 ...\n");
println("Loading Concrete dataset...\n");

concrete_df = CSV.File(eval(@__DIR__)*"/../data/Concrete_Data.csv") |> DataFrame;

transform!(concrete_df, "Concrete compressive strength (MPa)" => ByRow(strength -> get_category(strength)) => "Category")

open(eval(@__DIR__)*"/../data/ConcreteUCI.csv", "w") do file
    CSV.write(file, concrete_df)
end

strength_categories = Array(["Non-standard", "Standard", "High strength"]);
predictor_names = names(concrete_df)[1:end-2];
concrete_matrix = Matrix{Real}(concrete_df[:,predictor_names]);

display(concrete_df);

num_predictors   = length(predictor_names);
num_categories   = length(strength_categories);
num_observations = nrow(concrete_df);

predictors_statistics_df = DataFrame()

for predictor_name in predictor_names
    predictor_array = Array(concrete_df[:, predictor_name])
    append!(predictors_statistics_df, DataFrame(mean = mean(predictor_array), std = std(predictor_array), gamma = skewness(predictor_array)))
end
print(predictors_statistics_df)

predictors_corr_matrix = cor(concrete_matrix, concrete_matrix)

# pyplot()

# f = heatmap(predictor_names, predictor_names, predictors_corr_matrix,xtickfontrotation=20,framestyle=:box,clim=(-1,1),color=:balance,aspect_ratio=:equal);
# save_if_isfile(f,"predictors_corr_matrix.pdf");

# gr()
# for category in 0:num_categories
#     f = plot_monovariate_histograms(concrete_df, predictor_names, category)
#     save_if_isfile(f,"monovariate_histograms_$category.pdf");
# end

# for category in 0:num_categories
#     f = plot_bivariate_scatters(concrete_df, predictor_names, category)
#     save_if_isfile(f,"bivariate_scatters_$category.pdf");
# end