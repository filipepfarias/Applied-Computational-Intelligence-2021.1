using DataFrames, CSV, Plots, StatsBase, Statistics

concrete_df = dropmissing(CSV.File("./data/Concrete_Data.csv") |> DataFrame)

concrete_matrix = Matrix{Real}(concrete_df[:,:])
predictors_corr_matrix = cor(concrete_matrix, concrete_matrix)

labels = Array(names(concrete_df[:,:]))
Plots.heatmap(labels, labels, predictors_corr_matrix, xrotation = 45)

plot_matrix = Matrix{}(undef,9,9);

for i in 1:9
    for j in 1:9
        if i == j
            plot_matrix[i,j] = histogram(concrete_df[:,i])
        else
            plot_matrix[i,j] = scatter(concrete_df[:,i],concrete_df[:,j])
        end
    end    
end

f = plot(plot_matrix[:]..., layout=(9,9), size=(3000,3000),axis=false,ticks=false,legend=false)

savefig(f,"../figure/matrix-corplot.pdf");
