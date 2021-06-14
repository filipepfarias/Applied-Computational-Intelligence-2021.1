using Revise
using AppCompIntel 
using CSV
using DataFrames
using PrettyTables
using Statistics, StatsBase, StatsPlots
using Plots
using Latexify, LaTeXStrings
using MultivariateStats

println("Running HW1");
println("Loading Concrete dataset");

save_for_report = true;

concrete_df = CSV.File(eval(@__DIR__)*"/../data/Concrete_Data.csv") |> DataFrame;
transform!(concrete_df, "Concrete compressive strength (MPa)" => ByRow(strength -> get_category(strength)) => "Category");

strength_categories = Array(["Non-standard", "Standard", "High strength"]);
predictor_names = Array([L"D_1", L"D_2", L"D_3", L"D_4", L"D_5", L"D_6", L"D_7", L"D_8",]);
concrete_matrix = Matrix{Float64}(concrete_df[:,1:end-2]);
concrete_matrix = (concrete_matrix .- mean(concrete_matrix, dims=1)) ./ std(concrete_matrix, dims=1)

num_predictors   = length(predictor_names);
num_categories   = length(strength_categories);
num_observations = nrow(concrete_df);

println("Evaluating predictors statistics")

predictors_statistics_df              = DataFrame()
predictors_statistics_notstandard_df  = DataFrame()
predictors_statistics_standard_df     = DataFrame()
predictors_statistics_highstrength_df = DataFrame()

for predictor_idx in 1:num_predictors
    predictor_array = Array(concrete_df[:, predictor_idx])

    predictor_array_notstandard = Array(concrete_df[concrete_df[:, "Category"] .== 1, predictor_idx])
    predictor_array_standard = Array(concrete_df[concrete_df[:, "Category"] .== 2, predictor_idx])
    predictor_array_highstrength = Array(concrete_df[concrete_df[:, "Category"] .== 3, predictor_idx])

    append!(predictors_statistics_df, 
        DataFrame(mean = mean(predictor_array), std = std(predictor_array), gamma = skewness(predictor_array))
        );

    append!(predictors_statistics_notstandard_df, 
        DataFrame(mean = mean(predictor_array_notstandard), std = std(predictor_array_notstandard), gamma = skewness(predictor_array_notstandard))
        );

    append!(predictors_statistics_standard_df, 
        DataFrame(mean = mean(predictor_array_standard), std = std(predictor_array_standard), gamma = skewness(predictor_array_standard))
        );

    append!(predictors_statistics_highstrength_df, 
        DataFrame(mean = mean(predictor_array_highstrength), std = std(predictor_array_highstrength), gamma = skewness(predictor_array_highstrength))
        );
end

println("Class-unconditional predictors statistics");
println(predictors_statistics_df)

println("Class-unconditional predictors statistics");
println(predictors_statistics_notstandard_df)
println(predictors_statistics_standard_df)
println(predictors_statistics_highstrength_df)

println("Evaluating correlation matrix");
predictors_corr_matrix = cor(concrete_matrix, concrete_matrix)

pyplot()

f = heatmap(predictor_names, predictor_names, predictors_corr_matrix, 
    xrotation=20,framestyle=:box,clim=(-1,1),color=:balance,aspect_ratio=:equal);
!save_for_report ? display(f) : savefig(f,figure_path("predictors_corr_matrix.pdf"));

gr()

println("Plotting monovariate histograms")
f1 = @df concrete_df plot(cols(1:num_predictors), layout=grid(2,4), t = :histogram, bins = 8, 
    title = ["$i" for j in 1:1, i in predictor_names[:]], 
    titlefontsize= 14, tickfontsize=10, size = (1000,700),legend = false, lw = 0.1, framestyle = :box);
!save_for_report ? display(f1) : savefig(f1,figure_path("monovariate_histograms_allcategories.pdf"));

println("Plotting class-conditional monovariate histograms")
f2 = Array{Any}(undef,num_predictors);
for i = 1:num_predictors 
    f2[i] = @df concrete_df groupedhist(cols(i), group = :Category, bins=8, lw = 0.1, framestyle = :box)
end
f2 = plot(f2..., layout = grid(2,4), title = ["$i" for j in 1:1, i in predictor_names[:]], 
    titlefontsize= 14, tickfontsize=10, size = (1000,700));
!save_for_report ? display(f2) : savefig(f2,figure_path("monovariate_histograms_classcond.pdf"));

println("Plotting unconditional bivariate histograms");
f3 = Matrix{}(undef,8,8);
I  = LinearIndices(f3);

for i in 1:size(f3)[1]
    for j in 1:size(f3)[2]
        if i == j
            f3[i,i] = @df concrete_df groupedhist(cols(i), group = :Category, axis=true, ticks=false, legend=false, bins=10, lw = 0.1, framestyle = :box)
        else
            f3[i,j] = @df concrete_df scatter(cols(i), cols(j), group = :Category, axis=true, ticks=false, legend=false; markerstrokecolor=:white, markerstrokewidth=0.1, markersize=2.7, framestyle = :box)
        end
        
        j == 1 ? plot!(f3[I[i,j]], xguide=L"D_{%$i}",xmirror = true, xguideposition= :top) : plot!(f3[I[i,j]], top_margin=-2Plots.mm)
        i == 1 ? plot!(f3[I[i,j]], yguide=L"D_{%$j}", ymirror = true, yguideposition= :left) : plot!(f3[I[i,j]], left_margin=-2Plots.mm)
    end
end

f3 = plot(f3...,layout = grid(8,8), dpi=90, size=(800,800));
!save_for_report ? display(f3) : savefig(f3,figure_path("bivariate_histograms_allclass.pdf"));

println("Executing PCA");
concrete_pca              = MultivariateStats.fit(MultivariateStats.PCA, concrete_matrix', pratio=1);
concrete_pca_projection_matrix = MultivariateStats.projection(concrete_pca)
concrete_projected_matrix = MultivariateStats.transform(concrete_pca, concrete_matrix');

not_standard  = concrete_projected_matrix[:, concrete_df[:, "Category"] .== 1]
standard      = concrete_projected_matrix[:, concrete_df[:, "Category"] .== 2]
high_strength = concrete_projected_matrix[:, concrete_df[:, "Category"] .== 3]

f4 = scatter(not_standard[1,:], not_standard[2,:], marker=:circle,linewidth=0, label=L"not~standard", markerstrokecolor=:white, markerstrokewidth=0.5)
f4 = scatter!(standard[1,:], standard[2,:], marker=:circle, linewidth=0, label=L"standard", markerstrokecolor=:white, markerstrokewidth=0.5)
f4 = scatter!(high_strength[1,:], high_strength[2,:], marker=:circle, linewidth=0, label=L"high~strength", markerstrokecolor=:white, markerstrokewidth=0.5)
f4 = plot!(f4, xlabel=L"First~principal~component", ylabel=L"Second~principal~component", legend=:topleft, legendfontsize = 6, framestyle = :box);

for i in 1:num_predictors
    x = concrete_pca_projection_matrix[i,1]
    y = concrete_pca_projection_matrix[i,2]
    f4 = plot!(f4, [0, x],[0, y],arrow=true,color=:black,linewidth=1.5,label="")
    annotate!(x * 1.15, y * 1.15, text(i, :black, 5), :black)
end

!save_for_report ? display(f4) : savefig(f4,figure_path("pca_scatter_plot.pdf"));

f5 = plot(principalvars(concrete_pca), xlabel=L"Principal~components", ylabel=L"Variance", legend = false, framestyle = :box);
!save_for_report ? display(f5) : savefig(f6,figure_path("pca_variance.pdf"));

println("\nFinished!")
