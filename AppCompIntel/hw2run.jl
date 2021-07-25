# using Pkg
# Pkg.activate(".")

using Revise
using AppCompIntel 
using CSV
using DataFrames
using PrettyTables
using Statistics, StatsBase
using Plots
using Latexify, LaTeXStrings
using MultivariateStats
using MLJ

println("\nRunning HW2");
println("\nLoading Concrete dataset");

save_for_report = false;

concrete_df = CSV.File(eval(@__DIR__)*"/data/Concrete_Data.csv", normalizenames=true) |> DataFrame;
transform!(concrete_df, "Concrete_Compressive_Strength" => ByRow(strength -> get_category(strength)) => "Category");

println("\nRunning OLS Linear Regression");

results = DataFrame(["CV" => [], "RMSE" => [], "R²" => []]);

if save_for_report
    h1 = LatexHighlighter((data,i,j)->(data[i,j] == findmin(data[:, 2])[1]),
                            ["textbf"]);
    h2 = LatexHighlighter((data,i,j)->(data[i,j] == findmax(data[:, 3])[1]),
                        ["textbf"]);
else
    h1 = Highlighter((data,i,j)->(data[i,j] == findmin(data[:, 2])[1]),
                            bold       = true,
                            foreground = :cyan);
    h2 = Highlighter((data,i,j)->(data[i,j] == findmax(data[:, 3])[1]),
                        bold       = true,
                        foreground = :cyan); 
end

# Ingesting data
y, X = unpack(concrete_df, ==(:Concrete_Compressive_Strength), !=(:Category))

# Normalizing data
mapcols!(col -> (col .- mean(col))/std(col),X);
y = mapslices(col -> (col .- mean(col))/std(col),y; dims=1);

# Implementing R² Statistics
R²(ŷ, y) = 1 - sum((y.-ŷ).^2)/sum((y.-mean(y)).^2)

# Import Linear Regression
LinearRegressor = @load LinearRegressor pkg=MLJLinearModels verbosity=0

# Pipeline: Data is standardized and then goes through the linear regression
model_lr = LinearRegressor(
        fit_intercept = true,
        solver = nothing);

# Wraps model, predictor and outcome
model_lr_machine = machine(model_lr, X, y);

# Evaluate model performance for train set equals 70% of the total
model_lr_summary_70 = evaluate!(
    model_lr_machine,
    resampling=Holdout(fraction_train=0.5, rng=22),
    measure=[rmse, R²],
    verbosity=0);

push!(results,[
    "70% Train / 30% Test" model_lr_summary_70.measurement[1] model_lr_summary_70.measurement[2]
    ]);

# Evaluate model performance for 5-folds
model_lr_summary_kfolds = evaluate!(
    model_lr_machine,
    resampling=CV(nfolds=5, rng=930),
    measure=[rmse, R²],
    verbosity=0);

append!(results,
    DataFrame(
        "CV" => string.([1; 2; 3; 4; 5]).*"-th fold",
        "RMSE" =>  model_lr_summary_kfolds.per_fold[1],
        "R²" => model_lr_summary_kfolds.per_fold[2]
    ));

if save_for_report
    results = rename(results, "R²" => L"R^2");

    table = pretty_table(String, results; backend = Val(:latex),
        highlighters = (h1, h2),title="OLS Linear Regression");
    
    open("hw2/tables/results_lr.tex", "w") do io
        write(io, table)
    end; 
else
    pretty_table(results,highlighters = (h1, h2),title="OLS Linear Regression", crop = :none)
end

# Ridge Regression
println("\nRunning L²-penalised Linear (Ridge) Regression\n");

results = DataFrame(["CV" => [], "RMSE" => [], "R²" => []]);

RidgeRegressor = @load RidgeRegressor pkg=MLJLinearModels verbosity=0

model_rr = RidgeRegressor(
        fit_intercept = true,
        solver = nothing);

r = range(model_rr, :lambda, lower=1e-2, upper=100_000, scale=:log10);

tuned_model_rr = TunedModel(
    model=model_rr, 
    ranges=r, 
    tuning=Grid(resolution=50),
    resampling=CV(nfolds=5, rng=930), 
    measure=rmse)
   
# Wraps model, predictor and outcome
model_rr_machine = machine(tuned_model_rr, X, y);

# Evaluate model performance for train set equals 70% of the total
model_rr_summary_70 = evaluate!(
    model_rr_machine,
    resampling=Holdout(fraction_train=0.7, rng=930),
    measure=[rmse, R²],
    verbosity=0);

push!(results,[
    "70% Train / 30% Test" model_rr_summary_70.measurement[1] model_rr_summary_70.measurement[2]
    ]);

# Evaluate model performance for 5-folds
model_rr_summary_kfolds = evaluate!(
    model_rr_machine,
    resampling=CV(nfolds=5, rng=930),
    measure=[rmse, R²],
    verbosity=0);

append!(results,
    DataFrame(
        "CV" => string.([1; 2; 3; 4; 5]).*"-th fold",
        "RMSE" =>  model_rr_summary_kfolds.per_fold[1],
        "R²" => model_rr_summary_kfolds.per_fold[2]
    ));

if save_for_report
    results = rename(results, "R²" => L"R^2");

    table = pretty_table(String, results; backend = Val(:latex),
        highlighters = (h1, h2),title="Ridge Regression");

    open("hw2/tables/results_rr.tex", "w") do io
        write(io, table)
    end;
else
    pretty_table(results,highlighters = (h1, h2),title="Ridge Regression", crop = :none)
end

# PLS model
println("\nRunning Partial Least Squares Regression\n");

results = DataFrame(["CV" => [], "RMSE" => [], "R²" => []]);

PLSRegressor = @load PLSRegressor pkg=PartialLeastSquaresRegressor verbosity=0

model_pls = PLSRegressor();

r = range(model_pls, :n_factors, lower=1, upper=8, scale=:linear);

tuned_model_pls = TunedModel(
    model=model_pls, 
    ranges=r, 
    tuning=Grid(resolution=8),
    resampling=CV(nfolds=5, rng=930), 
    measure=rmse);

# Wraps model, predictor and outcome
model_pls_machine = machine(tuned_model_pls, X, y);

# Evaluate model performance for train set equals 70% of the total
model_pls_summary_70 = evaluate!(
    model_pls_machine,
    resampling=Holdout(fraction_train=0.7, rng=930),
    measure=[rmse, R²],
    verbosity=0);

push!(results,[
    "70% Train / 30% Test" model_pls_summary_70.measurement[1] model_pls_summary_70.measurement[2]
    ]);

model_pls_summary_kfolds = evaluate!(
    model_pls_machine,
    resampling=CV(nfolds=5, rng=930),
    measure=[rmse, R²],
    verbosity=0);

append!(results,
    DataFrame(
        "CV" => string.([1; 2; 3; 4; 5]).*"-th fold",
        "RMSE" =>  model_pls_summary_kfolds.per_fold[1],
        "R²" => model_pls_summary_kfolds.per_fold[2]
    ));

if save_for_report
    results = rename(results, "R²" => L"R^2");

    table = pretty_table(String, results; backend = Val(:latex),
        highlighters = (h1, h2),title="PLS Regression");

    open("hw2/tables/results_pls.tex", "w") do io
        write(io, table)
    end;
else
    pretty_table(results,highlighters = (h1, h2),title="\nPLS Regression", crop = :none)
end

# Plots
if save_for_report
    # Correlation + Scatter + Histogram
    figure = scatterplot(select(concrete_df, Not(:Category)))
    savefig(figure,figure_path("correlation_predictors_outcomes.pdf"))

    figure = plot(map(p -> p.second,
        model_lr_summary_70.fitted_params_per_fold[1].coefs), 
        marker = :circle, markerstrokewidth=0.3 ,
        xticks = (1:8, latexstring.("D_" .* string.(1:8))), 
        label = "Linear Regression")
    figure = plot!(map(p -> p.second,
        model_rr_summary_70.fitted_params_per_fold[1].best_fitted_params[1]), 
        marker = :diamond, markerstrokewidth=0.3, 
        label = "L₂-Penalised",
        size=(360,180))
    savefig(figure,figure_path("fitted_params_70.pdf"))
    
    figure = plot(map(p -> p.second,
        model_lr_summary_kfolds.fitted_params_per_fold[end].coefs), 
        marker = :circle, markerstrokewidth=0.3, 
        xticks = (1:8, latexstring.("D_" .* string.(1:8))), 
        label = "Linear Regression")
    figure = plot!(map(p -> p.second,
        model_rr_summary_kfolds.fitted_params_per_fold[1].best_fitted_params[1]), 
        marker = :diamond, markerstrokewidth=0.3, 
        label = "L₂-Penalised",
        size=(360,180))
    savefig(figure,figure_path("fitted_params_kfolds.pdf"))
end

# To add:

# Variance explained

# P = model_pls_summary_kfolds.fitted_params_per_fold[1][2][1].P
# DataFrame("Principal Components" => "PC".*string.(1:8), "Variance Explained (%)" => round.((eigen(P*P').values) ./sum(eigen(P*P').values) * 100; digits=2 )[end:-1:1])