module AppCompIntel

    using Plots
    using StatsPlots
    using Statistics: cor
    using DataFrames
    using LaTeXStrings

    export
        #Utils - Plots
        figure_path,
        get_category,
        scatterplot

    include("utils.jl");

end # module
