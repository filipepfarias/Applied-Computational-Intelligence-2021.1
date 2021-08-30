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
        "RMSE" =>  model_nnr_summary_kfolds.per_fold[1],
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
    pretty_table(results,highlighters = (h1, h2),title="Neural Network Regression", crop = :none, nosubheader=true)
end

# Ingesting data FOR CLASSIFICATION
y, X = unpack(concrete_df,
                ==(:Category), 
                !=(:Concrete_Compressive_Strength);
                :Category => Multiclass,
                :Age => Continuous);

if true
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

# Evaluate model performance for 5-folds
model_lda_summary_kfolds = evaluate!(
    model_lda_machine,
    resampling=CV(nfolds=5, rng=930),
    measure=ConfusionMatrix(perm=[1,2,3]),
    operation=predict_mode
    );

results = model_lda_summary_kfolds.measurement[1];
if save_for_report
    table = pretty_table(String, results.mat; backend = Val(:latex), header = results.labels,
        title="Linear Discriminant Analysis Classification", nosubheader=true);
    
    open("hw3/tables/results_lda.tex", "w") do io
        write(io, table)
    end; 
else
    pretty_table(results.mat,title="Linear Discriminant Analysis Classification", crop = :none, nosubheader=true, header = results.labels)
end

println("\nRunning Neural Networks Classification");

# Import Neural Networks Classifier
NNClassifier = @load NeuralNetworkClassifier pkg=MLJFlux verbosity=0

model_nnc = NNClassifier();

# Wraps model, predictor and outcome
model_nnc_machine = machine(model_nnc, X, y);

# Evaluate model performance for 5-folds
model_nnc_summary_kfolds = evaluate!(
    model_nnc_machine,
    resampling=CV(nfolds=5, rng=930),
    measure=ConfusionMatrix(perm=[1,2,3]),
    operation=predict_mode
    );

results = model_nnc_summary_kfolds.measurement[1];
if save_for_report
    table = pretty_table(String, results.mat; backend = Val(:latex), header = results.labels,
        title="Neural Networks Classification", nosubheader=true);
    
    open("hw3/tables/results_nnc.tex", "w") do io
        write(io, table)
    end; 
else
    pretty_table(results.mat,title="Neural Networks Classification", crop = :none, nosubheader=true, header = results.labels)
end

println("\nRunning Nearest Neighbors Classification");


# Import Linea Discriminant Analysis
KNNClassifier = @load KNNClassifier pkg=NearestNeighborModels verbosity=0

model_knnc = KNNClassifier();

# Wraps model, predictor and outcome
model_knnc_machine = machine(model_knnc, X, y);

# Evaluate model performance for 5-folds
model_knnc_summary_kfolds = evaluate!(
    model_knnc_machine,
    resampling=CV(nfolds=5, rng=930),
    measure=ConfusionMatrix(perm=[1,2,3]),
    operation=predict_mode
    );

results = model_knnc_summary_kfolds.measurement[1];
if save_for_report
    table = pretty_table(String, results.mat; backend = Val(:latex), header = results.labels,
        title="Nearest Neighbors Classification", nosubheader=true);
    
    open("hw3/tables/results_knnc.tex", "w") do io
        write(io, table)
    end; 
else
    pretty_table(results.mat,title="Nearest Neighbors Classification", crop = :none, nosubheader=true, header = results.labels)
end

println("\nRunning Support Vector Classification");

# Import Linea Discriminant Analysis
SVCClassifier = @load SVC pkg=LIBSVM verbosity=0

model_svc = SVCClassifier();

# Wraps model, predictor and outcome
model_svc_machine = machine(model_svc, X, y);

# Evaluate model performance for 5-folds
model_svc_summary_kfolds = evaluate!(
    model_svc_machine,
    resampling=CV(nfolds=5, rng=930),
    measure=ConfusionMatrix(perm=[1,2,3])
    );

results = model_svc_summary_kfolds.measurement[1];
if save_for_report
    table = pretty_table(String, results.mat; backend = Val(:latex), header = results.labels,
        title="Support Vector Classification", nosubheader=true);
    
    open("hw3/tables/results_svc.tex", "w") do io
        write(io, table)
    end; 
else
    pretty_table(results.mat,title="Support Vector Classification", crop = :none, nosubheader=true, header = results.labels)
end