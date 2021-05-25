using DataFrames, CSV, Plots, StatsBase, Missings, RDatasets
air_quality_df = dropmissing(CSV.File("./data/AirQualityUCI.csv", delim=";") |> DataFrame);