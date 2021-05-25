using DataFrames, CSV, Plots, StatsBase, Statistics

air_quality_df = dropmissing(CSV.File("./data/AirQualityUCI.csv", delim=";", missingstring="-200") |> DataFrame)

predictor_names = Array(["CO_GT", "PT08_S1_CO", "NMHC_GT", "C6H6_GT", "PT08_S2_NMHC", "NOx_GT", "PT08_S3_NOx", "NO2_GT", "PT08_S4_NO2", "PT08_S5_O3", "RH", "AH"])

air_quality_matrix = Matrix{Real}(air_quality_df[:, predictor_names])

predictors_corr_matrix = cor(air_quality_matrix, air_quality_matrix)

Plots.heatmap(predictor_names, predictor_names, predictors_corr_matrix, size=(1500,1000))