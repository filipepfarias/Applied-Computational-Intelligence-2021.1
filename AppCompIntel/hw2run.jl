# using Core: Matrix
using Pkg
Pkg.activate(".")

using Revise
using AppCompIntel 
using CSV
using DataFrames
using PrettyTables
using Statistics, StatsBase, StatsPlots
using Plots
using Latexify, LaTeXStrings
using MultivariateStats
using MLDataUtils

println("\nRunning HW2");
println("\nLoading Concrete dataset\n");

save_for_report = false;

concrete_df = CSV.File(eval(@__DIR__)*"/data/Concrete_Data.csv", normalizenames=true) |> DataFrame;
transform!(concrete_df, "Concrete_Compressive_Strength" => ByRow(strength -> get_category(strength)) => "Category");
train_df, test_df = splitobs(shuffleobs(concrete_df), at = 0.7)
concrete_folds = kfolds(concrete_df, k = 5)

strength_categories = Array(["Non-standard", "Standard", "High strength"]);
predictors_outcome_names = Array([L"D_1", L"D_2", L"D_3", L"D_4", L"D_5", L"D_6", L"D_7", L"D_8", L"Y"]);

num_predictors   = length(predictors_outcome_names) - 1;
num_categories   = length(strength_categories);
num_observations = nrow(concrete_df);
num_train_observations = nrow(train_df);
num_test_observations = nrow(test_df);

concrete_matrix = Matrix{Float64}(concrete_df[:,1:end-1]);
concrete_matrix = (concrete_matrix .- mean(concrete_matrix, dims=1)) ./ std(concrete_matrix, dims=1);

# pyplot()

# println("\nEvaluating correlation matrix\n");
# predictors_corr_matrix = cor(concrete_matrix, concrete_matrix)

# f = heatmap(predictors_outcome_names, predictors_outcome_names, predictors_corr_matrix, 
#     xrotation=20,framestyle=:box,clim=(-1,1),color=:balance,aspect_ratio=:equal);
# !save_for_report ? display(f) : savefig(f,figure_path("predictors_corr_matrix.pdf"));


train_matrix = Matrix{Float64}(train_df[:,1:end-1]);
train_matrix = (train_matrix .- mean(train_matrix, dims=1)) ./ std(train_matrix, dims=1);
train_predictors = train_matrix[:, 1:end-1];
train_outcome = train_matrix[:, end];

test_matrix = Matrix{Float64}(test_df[:,1:end-1]);
test_matrix = (test_matrix .- mean(test_matrix, dims=1)) ./ std(test_matrix, dims=1)
test_predictors = test_matrix[:, 1:end-1];
test_outcome = test_matrix[:, end];

# solve using llsq
train_coefficients = llsq(train_predictors, train_outcome;)

# do prediction
train_outcome_prediction = train_predictors * train_coefficients[1:end-1] .+ train_coefficients[end]

# measure the error
train_rmse = sqrt(mean(abs2.(train_outcome .- train_outcome_prediction)))
train_rss = sum((train_outcome .- train_outcome_prediction).^2)
train_tss = sum((train_outcome .- mean(train_outcome)).^2)
train_r2 = 1 - train_rss / train_tss

print("train_rmse = $train_rmse")
print("\ntrain_r2 = $train_r2\n")

# do prediction in test
test_outcome_prediction = test_predictors * train_coefficients[1:end-1] .+ train_coefficients[end]

# measure the error
test_rmse = sqrt(mean(abs2.(test_outcome .- test_outcome_prediction)))
test_rss = sum((test_outcome .- test_outcome_prediction).^2)
test_tss = sum((test_outcome .- mean(test_outcome)).^2)
test_r2 = 1 - test_rss / test_tss

print("\ntest_rmse = $test_rmse")
print("\ntest_r2 = $test_r2\n")

for k in [1, 2, 3, 4, 5]
    (kfold_train, kfold_test) = concrete_folds[k]
    kfold_train = Matrix{Float64}(kfold_train)
    kfold_train_predictors = kfold_train[:, 1:end-1]
    kfold_train_outcome = kfold_train[:, end]

    kfold_test =  Matrix{Float64}(kfold_test)
    kfold_test_predictors = kfold_test[:, 1:end-1]
    kfold_test_outcome = kfold_test[:, end]

    # solve using llsq
    kfold_train_coefficients = llsq(kfold_train_predictors, kfold_train_outcome)
    # do prediction train
    kfold_train_outcome_prediction = kfold_train_predictors * kfold_train_coefficients[1:end-1] .+ kfold_train_coefficients[end]
    # measure the error
    kfold_train_rmse = sqrt(mean(abs2.(kfold_train_outcome .- kfold_train_outcome_prediction)))
    kfold_train_rss = sum((kfold_train_outcome .- kfold_train_outcome_prediction).^2)
    kfold_train_tss = sum((kfold_train_outcome .- mean(kfold_train_outcome)).^2)
    kfold_train_r2 = 1 - kfold_train_rss / kfold_train_tss

    # do prediction train
    kfold_test_outcome_prediction = kfold_test_predictors * kfold_train_coefficients[1:end-1] .+ kfold_train_coefficients[end]
    # measure the error
    kfold_test_rmse = sqrt(mean(abs2.(kfold_test_outcome .- kfold_test_outcome_prediction)))
    kfold_test_rss = sum((kfold_test_outcome .- kfold_test_outcome_prediction).^2)
    kfold_test_tss = sum((kfold_test_outcome .- mean(kfold_test_outcome)).^2)
    kfold_test_r2 = 1 - kfold_test_rss / kfold_test_tss

    print("\nkfold run number $k")
    print("\nkfold_train_rmse = $kfold_train_rmse")
    print("\nkfold_train_r2 = $kfold_train_r2\n")

    print("\nkfold_test_rmse = $kfold_test_rmse")
    print("\nkfold_test_r2 = $kfold_test_r2\n\n")
end