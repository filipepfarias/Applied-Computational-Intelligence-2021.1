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

<<<<<<< HEAD
h1 = Highlighter((data,i,j)->(data[i,j] == findmin(data[:, 2])[1]),
                         bold       = true,
                         foreground = :cyan);
h2 = Highlighter((data,i,j)->(data[i,j] == findmin(data[:, 3])[1]),
                         bold       = true,
                         foreground = :cyan);
pretty_table(results,highlighters = (h1, h2),title="Least Squares Regression")

results = DataFrame(["Fold" => [], "RMSE" => [], "R^2" => []]);
# solve using ridge
# do prediction
λ = exp.(-100:1:100)
log_train_rmse = [];
for λ in exp.(-100:1:100)
    train_coefficients = ridge(train_predictors, train_outcome,λ)
    train_outcome_prediction = [train_predictors ones(size(train_predictors)[1],1)]*train_coefficients
    train_rmse = sqrt(mean(abs2.(train_outcome .- train_outcome_prediction)))
    push!(log_train_rmse,train_rmse)
end

# find the minimal error for λ
(_,id) = findmin(log_train_rmse)
λop = λ[id]
train_coefficients = ridge(train_predictors, train_outcome, λop)

# measure the error
train_rmse = sqrt(mean(abs2.(train_outcome .- train_outcome_prediction)))
train_rss = sum((train_outcome .- train_outcome_prediction).^2)
train_tss = sum((train_outcome .- mean(train_outcome)).^2)
train_r2 = 1 - train_rss / train_tss

# do prediction in test
test_outcome_prediction = test_predictors * train_coefficients[1:end-1] .+ train_coefficients[end]

# measure the error
test_rmse = sqrt(mean(abs2.(test_outcome .- test_outcome_prediction)))
test_rss = sum((test_outcome .- test_outcome_prediction).^2)
test_tss = sum((test_outcome .- mean(test_outcome)).^2)
test_r2 = 1 - test_rss / test_tss

push!(results,["70% Train / 30% Test" test_rmse test_r2]);

for k in [1, 2, 3, 4, 5]
    (kfold_train, kfold_test) = concrete_folds[k]
    kfold_train = Matrix{Float64}(kfold_train)
    kfold_train_predictors = kfold_train[:, 1:end-1]
    kfold_train_outcome = kfold_train[:, end]

    kfold_test =  Matrix{Float64}(kfold_test)
    kfold_test_predictors = kfold_test[:, 1:end-1]
    kfold_test_outcome = kfold_test[:, end]

    # solve using ridge
    λ = exp.(-100:1:100)
    log_train_rmse = [];
    for λi in λ
        kfold_train_coefficients = ridge(kfold_train_predictors, kfold_train_predictors, λi)
        kfold_train_outcome_prediction = [kfold_train_predictors ones(size(kfold_train_predictors)[1],1)]*kfold_train_coefficients
        train_rmse = sqrt(mean(abs2.(kfold_train_predictors .- kfold_train_outcome_prediction)))
        push!(log_train_rmse,train_rmse)
    end

    # find the minimal error for λ
    (_,id) = findmin(log_train_rmse)
    λop = λ[id]
    kfold_train_coefficients = ridge(kfold_train_predictors, kfold_train_outcome, λop)

    # do prediction train
    kfold_train_outcome_prediction = kfold_train_predictors * kfold_train_coefficients[1:end-1] .+ kfold_train_coefficients[end]
    # measure the error
    # kfold_train_rmse = sqrt(mean(abs2.(kfold_train_outcome .- kfold_train_outcome_prediction)))
    # kfold_train_rss = sum((kfold_train_outcome .- kfold_train_outcome_prediction).^2)
    # kfold_train_tss = sum((kfold_train_outcome .- mean(kfold_train_outcome)).^2)
    # kfold_train_r2 = 1 - kfold_train_rss / kfold_train_tss

    # do prediction train
    kfold_test_outcome_prediction = kfold_test_predictors * kfold_train_coefficients[1:end-1] .+ kfold_train_coefficients[end]
    # measure the error
    kfold_test_rmse = sqrt(mean(abs2.(kfold_test_outcome .- kfold_test_outcome_prediction)))
    kfold_test_rss = sum((kfold_test_outcome .- kfold_test_outcome_prediction).^2)
    kfold_test_tss = sum((kfold_test_outcome .- mean(kfold_test_outcome)).^2)
    kfold_test_r2 = 1 - kfold_test_rss / kfold_test_tss

    push!(results,[string(k)*"-fold" kfold_test_rmse kfold_test_r2])
end
=======
println("\nRunning OLS Linear Regression");

results = DataFrame(["CV" => [], "RMSE" => [], "R²" => []]);
>>>>>>> 3cfc724... OLS with MLJ implemented

h1 = Highlighter((data,i,j)->(data[i,j] == findmin(data[:, 2])[1]),
                        bold       = true,
                        foreground = :cyan);
h2 = Highlighter((data,i,j)->(data[i,j] == findmax(data[:, 3])[1]),
                        bold       = true,
                        foreground = :cyan); 

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

pretty_table(results,highlighters = (h1, h2),title="OLS Linear Regression")

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

pretty_table(results,highlighters = (h1, h2),title="Ridge Regression")

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

pretty_table(results,highlighters = (h1, h2),title="\nPLS Regression")