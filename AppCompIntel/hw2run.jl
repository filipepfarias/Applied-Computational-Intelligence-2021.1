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
X = StatsBase.transform(StatsBase.fit(ZScoreTransform, Matrix(X); dims = 1),Matrix(X));
y = StatsBase.transform(StatsBase.fit(ZScoreTransform, y; dims = 1),y);

# Implementing R² Statistics
R²(ŷ, y) = 1 - sum((y.-ŷ).^2)/sum((y.-mean(y)).^2)

# Import Linear Regression
LinearRegressor = @load LinearRegressor pkg=MLJLinearModels verbosity=0

# Pipeline: Data is standardized and then goes through the linear regression
model = @pipeline(
    LinearRegressor(
        fit_intercept = true,
        solver = nothing)
        );

# Wraps model, predictor and outcome
model_machine = machine(model, X, y);

# Evaluate model performance for train set equals 70% of the total
model_summary_70 = evaluate!(
    model_machine,
    resampling=Holdout(fraction_train=0.5, rng=22),
    measure=[rmse, R²],
    verbosity=0);

push!(results,[
    "70% Train / 30% Test" model_summary_70.measurement[1] model_summary_70.measurement[2]
    ]);

model_summary_kfolds = evaluate!(
    model_machine,
    resampling=CV(nfolds=5, rng=443),
    measure=[rmse, R²],
    verbosity=0);

append!(results,
    DataFrame(
        "CV" => string.([1; 2; 3; 4; 5]).*"-th fold",
        "RMSE" =>  model_summary_kfolds.per_fold[1],
        "R²" => model_summary_kfolds.per_fold[2]
    ));
  
pretty_table(results,highlighters = (h1, h2),title="OLS Linear Regression")