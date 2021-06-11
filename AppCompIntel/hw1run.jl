using Base: Float64
using Revise
using AppCompIntel 
using CSV
using DataFrames
using PrettyTables
using Statistics, StatsBase, StatsPlots
using Plots
using Latexify, LaTeXStrings
using YeoJohnsonTrans
using BoxCoxTrans
using MultivariateStats

println("Running HW1 ...\n");
println("Loading Concrete dataset...\n");
pgfplotsx()

save_for_report = false;

concrete_df = CSV.File(eval(@__DIR__)*"/../data/Concrete_Data.csv") |> DataFrame;
#filter!(col -> !any(value -> value == 0.0, col), concrete_df);
transform!(concrete_df, "Concrete compressive strength (MPa)" => ByRow(strength -> get_category(strength)) => "Category");


# open(eval(@__DIR__)*"/../data/ConcreteUCI.csv", "w") do file
#     CSV.write(file, concrete_df)
# end

<<<<<<< HEAD
strength_categories = Array(["Non-standard", "Standard", "High strength"]);
predictor_names = names(concrete_df)[1:end-2];
concrete_matrix = Matrix{Real}(concrete_df[:,predictor_names]);
=======
strength_categories = Array(["not standard", "standard", "high strength"]);
predictor_names = Array(["Cement", "Blast Furnace Slag", "Fly Ash", "Water", "Superplasticizer", "Coarse Aggregate", "Fine Aggregate", "Age (day)"]);
>>>>>>> origin/dev

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

println("Evaluating predictors statistics...\n")
predictors_statistics_df = DataFrame()
<<<<<<< HEAD
for predictor_name in predictor_names
    predictor_array = Array(concrete_df[:, predictor_name])
    append!(predictors_statistics_df, DataFrame(
        mean = mean(predictor_array), std = std(predictor_array), gamma = skewness(predictor_array)
        ));
end
insertcols!(predictors_statistics_df,1,:Predictors => predictor_names);
display(predictors_statistics_df)

println("Calculating correlation matrix and plotting...\n")
pyplot()
predictors_corr_matrix = cor(concrete_matrix, concrete_matrix)
=======
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
>>>>>>> origin/dev

gr()
f = heatmap(predictor_names, predictor_names, predictors_corr_matrix, 
    xtickfontrotation=20,framestyle=:box,clim=(-1,1),color=:balance,aspect_ratio=:equal);
!save_for_report ? nothing : savefig(f,figure_path("predictors_corr_matrix.pdf"));

println("Plotting monovariate histograms...\n")
f1 = @df concrete_df plot(cols(1:num_predictors), layout=grid(2,4), t = :histogram, bins = 8, 
    title = ["$i" for j in 1:1, i in predictor_names[:]], 
    titlefontsize= 14, tickfontsize=10, size = (1000,1000),legend = false);
display(f1);
!save_for_report ? nothing : savefig(f1,figure_path("monovariate_histograms_allcategories.pdf"));

println("Plotting class-conditional monovariate histograms...\n")

f2 = Array{Any}(undef,num_predictors);
for i = 1:num_predictors 
    f2[i] = @df concrete_df groupedhist(cols(i), group = :Category, bins=8)
end
f2 = plot(f2..., layout = grid(2,4), title = ["$i" for j in 1:1, i in predictor_names[:]], 
    titlefontsize= 14, tickfontsize=10, size = (1000,1000));
display(f2);
!save_for_report ? nothing : savefig(f2,figure_path("monovariate_histograms_classcond.pdf"));

println("Plotting unconditional bivariate histograms...\n")

f3 = Matrix{}(undef,8,8);
I = LinearIndices(f3);

for i in 1:size(f3)[1]
    for j in 1:size(f3)[2]
        if i == j
            f3[i,i] = histogram(concrete_df[:,i], axis=false, ticks=false, legend=false, bins=10)
        else
            f3[i,j] = scatter(concrete_df[:,i],concrete_df[:,j],axis=false, ticks=false, legend=false; markerstrokewidth=.35, markersize=2)
        end
        
        j == 1 ? plot!(f3[I[i,j]], xguide=L"D_{%$i}",xmirror = true, xguideposition= :top) : plot!(f3[I[i,j]], top_margin=-2Plots.mm)
        i == 1 ? plot!(f3[I[i,j]], yguide=L"D_{%$j}", ymirror = true, yguideposition= :left) : plot!(f3[I[i,j]], left_margin=-2Plots.mm)
    end
end

<<<<<<< HEAD
f3 = plot(f3...,layout = grid(8,8), dpi=170, size=(700,700));
display(f3);
!save_for_report ? nothing : savefig(f3,figure_path("bivariate_histograms_allclass.pdf"));

# f = plot_monovariate_histograms_transf(concrete_df, predictor_names)
# save_if_isfile(f,"monovariate_histograms_transf.pdf");
=======
# for category in 0:num_categories
#     f = plot_monovariate_histograms(concrete_df, predictor_names, category, false)
#     save_if_isfile(f,"monovariate_histograms_$category.pdf");
# end

# for category in 0:0
#     f = plot_bivariate_scatters(concrete_df, predictor_names, category)
#     save_if_isfile(f,"bivariate_scatters_$category.pdf");
# end
>>>>>>> origin/dev
