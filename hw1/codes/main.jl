using DataFrames, CSV, Plots, StatsBase, Statistics

air_quality_df = dropmissing(CSV.File("./data/AirQualityUCI.csv", delim=";", missingstring="-200") |> DataFrame)

air_quality_matrix = Matrix{Real}(air_quality_df[:, 3:end])
predictors_corr_matrix = cor(air_quality_matrix, air_quality_matrix)

labels = Array(names(air_quality_df[:, 3:end]))
Plots.heatmap(labels, labels, predictors_corr_matrix, size=(1500,1000))