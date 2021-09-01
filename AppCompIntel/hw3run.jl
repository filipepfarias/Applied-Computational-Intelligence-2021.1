# using Pkg
# Pkg.activate(".")

using Revise
using AppCompIntel 
using CSV
using DataFrames
using PrettyTables
using Statistics, StatsBase, StatsPlots
using Plots
using Latexify, LaTeXStrings
using MultivariateStats
using MLJBase, MLJ, MLJFlux
using Flux

println("\nRunning HW3");
println("\nLoading Concrete dataset");

save_for_report = false;

concrete_df = CSV.File(eval(@__DIR__)*"/data/Concrete_Data.csv", normalizenames=true) |> DataFrame;
transform!(concrete_df, "Concrete_Compressive_Strength" => ByRow(strength -> get_category(strength)) => "Category");

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

println("\nRunning Neural Network Regression");

results = DataFrame(["CV" => [], "RMSE" => [], "R²" => []]);

# Ingesting data FOR REGRESSION
y, X = unpack(concrete_df, ==(:Concrete_Compressive_Strength), !=(:Category))

y_mean,y_std = mean(y),std(y)

# Normalizing data
mapcols!(col -> (col .- mean(col))/std(col),X);
y = mapslices(col -> (col .- mean(col))/std(col),y; dims=1);                

# Implementing R² Statistics
R²(ŷ, y) = 1 - sum((y.-ŷ).^2)/sum((y.-mean(y)).^2)

# Import Neural Network Regression
NNRegressor = @load NeuralNetworkRegressor pkg=MLJFlux verbosity=0

model_nnr = NNRegressor(
    builder = MLJFlux.Linear(σ=Flux.sigmoid)
);

# Wraps model, predictor and outcome
model_nnr_machine = machine(model_nnr, X, y);

# Evaluate model performance for 5-folds
model_nnr_summary_kfolds = evaluate!(
    model_nnr_machine,
    resampling=CV(nfolds=5, rng=930),
    measure=[rmse, R²],
    verbosity=0
    );

append!(results,
    DataFrame(
        "CV" => string.([1; 2; 3; 4; 5]).*"-th fold",
        "RMSE" =>  (model_nnr_summary_kfolds.per_fold[1] .+ y_mean)*y_std,
        "R²" => model_nnr_summary_kfolds.per_fold[2]
    ));

if save_for_report
    results = rename(results, "R²" => L"R^2");

    table = pretty_table(String, results; backend = Val(:latex),
        highlighters = (h1, h2),title="Neural Network Regression", nosubheader=true);
    
    open("hw3/tables/results_nnr.tex", "w") do io
        write(io, table)
    end; 
else
    println("\nFor E[y] = "*string(y_mean)*" and Var[y] = "*string(y_std))
    pretty_table(results,highlighters = (h1, h2),title="Neural Network Regression", crop = :none, nosubheader=true)
end

# Ingesting data FOR CLASSIFICATION
y, X = unpack(concrete_df,
                ==(:Category), 
                !=(:Concrete_Compressive_Strength);
                :Category => Multiclass,
                :Age => Continuous);

if false
    f1 = @df concrete_df boxplot(:Category,cols([1]), title="Cement", legend=false)
    f2 = @df concrete_df boxplot(:Category,cols([5]), title="Superplasticizer", legend=false)
    f3 = @df concrete_df boxplot(:Category,cols([8]), title="Age", legend=false)
    f  = plot(f1,f2,f3; layout=grid(1,3),size=(680,360))
    savefig(f,"hw3/figures/most_corr_predictors.pdf");
end

println("\nRunning Linear Discriminant Analysis Classification");

# Import Linear Discriminant Analysis
LDAClassifier = @load LDA pkg=MultivariateStats verbosity=0

model_lda = LDAClassifier();

# Wraps model, predictor and outcome
model_lda_machine = machine(model_lda, X, y);

results = DataFrame(["Class" => [], "Accuracy" => [], "Precision" => [], "Recall" => [], "F1 score" => []]);

fit!(model_lda_machine, verbosity=0)

ŷ = MLJBase.predict(model_lda_machine);
ŷ_label = mode.(ŷ);

for class in ["L1","L2","L3"]
    TP = MLJBase.multiclass_true_positive(mode.(ŷ),y)[class];
    FP = MLJBase.multiclass_false_positive(mode.(ŷ),y)[class];
    TN = MLJBase.multiclass_true_negative(mode.(ŷ),y)[class];
    FN = MLJBase.multiclass_false_negative(mode.(ŷ),y)[class];
    append!(
        results,
        DataFrame(["Class" => [class],
                   "Accuracy" => [(TP+TN)/(TP+TN+FP+FN)], 
                   "Precision" => [TP/(TP+FP)], 
                   "Recall" => [TP/(TP+FN)], 
                   "F1 score" => [2*TP/(2*TP+FP+FN)]])
        );
end

if save_for_report
    table = pretty_table(String, results.mat; backend = Val(:latex),
        title="Linear Discriminant Analysis Classification", nosubheader=true);
    
    # open("hw3/tables/results_lda.tex", "w") do io
    #     write(io, table)
    # end; 
else
    pretty_table(results,title="Linear Discriminant Analysis Classification", crop = :none, nosubheader=true)
    ConfusionMatrix(perm=[1,2,3])(mode.(ŷ),y)
end

println("\nRunning Nearest Neighbors Classification");


# Import k-Nearest Neighbors
KNNClassifier = @load KNNClassifier pkg=NearestNeighborModels verbosity=0

model_knnc = KNNClassifier();

K_range = range(model_knnc, :K, lower=1, upper=21);

# Evaluate model performance for 5-folds
self_tuning_knn = TunedModel(model=model_knnc,
                             resampling = CV(nfolds=5, rng=930),
                             tuning = Grid(resolution=5),
                             range = K_range,
                             measure=MisclassificationRate(),
                             operation=predict_mode);

# Wraps model, predictor and outcome
model_knnc_machine = machine(self_tuning_knn, X, y);

results = DataFrame(["Class" => [], "Accuracy" => [], "Precision" => [], "Recall" => [], "F1 score" => []]);

fit!(model_knnc_machine, verbosity=0)

ŷ = MLJBase.predict(model_knnc_machine);
ŷ_label = mode.(ŷ)

for class in ["L1","L2","L3"]
    TP = MLJBase.multiclass_true_positive(mode.(ŷ),y)[class];
    FP = MLJBase.multiclass_false_positive(mode.(ŷ),y)[class];
    TN = MLJBase.multiclass_true_negative(mode.(ŷ),y)[class];
    FN = MLJBase.multiclass_false_negative(mode.(ŷ),y)[class];
    append!(
        results,
        DataFrame(["Class" => [class],
                   "Accuracy" => [(TP+TN)/(TP+TN+FP+FN)], 
                   "Precision" => [TP/(TP+FP)], 
                   "Recall" => [TP/(TP+FN)], 
                   "F1 score" => [2*TP/(2*TP+FP+FN)]])
        );
end

if save_for_report
    table = pretty_table(String, results.mat; backend = Val(:latex), header = results.labels,
        title="Nearest Neighbors Classification", nosubheader=true);
    
    # open("hw3/tables/results_knnc.tex", "w") do io
    #     write(io, table)
    # end; 
else
    pretty_table(results,title="Nearest Neighbors Classification", crop = :none, nosubheader=true)
    ConfusionMatrix(perm=[1,2,3])(mode.(ŷ),y)
end

println("\nRunning Support Vector Classification");

# Import Support Vector Classification
SVCClassifier = @load SVC pkg=LIBSVM verbosity=0

model_svc = SVCClassifier();

# Wraps model, predictor and outcome
model_svc_machine = machine(model_svc, X, y);

results = DataFrame(["Class" => [], "Accuracy" => [], "Precision" => [], "Recall" => [], "F1 score" => []]);

fit!(model_svc_machine, verbosity=0);

ŷ_label = MLJBase.predict(model_svc_machine);

for class in ["L1","L2","L3"]
    TP = MLJBase.multiclass_true_positive(ŷ,y)[class];
    FP = MLJBase.multiclass_false_positive(ŷ,y)[class];
    TN = MLJBase.multiclass_true_negative(ŷ,y)[class];
    FN = MLJBase.multiclass_false_negative(ŷ,y)[class];
    append!(
        results,
        DataFrame(["Class" => [class],
                   "Accuracy" => [(TP+TN)/(TP+TN+FP+FN)], 
                   "Precision" => [TP/(TP+FP)], 
                   "Recall" => [TP/(TP+FN)], 
                   "F1 score" => [2*TP/(2*TP+FP+FN)]])
        );
end

if save_for_report
    table = pretty_table(String, results.mat; backend = Val(:latex),
        title="Support Vector Classification", nosubheader=true);
    
    # open("hw3/tables/results_knnc.tex", "w") do io
    #     write(io, table)
    # end; 
else
    pretty_table(results,title="Support Vector Classification", crop = :none, nosubheader=true)
    ConfusionMatrix(perm=[1,2,3])(ŷ,y)
end