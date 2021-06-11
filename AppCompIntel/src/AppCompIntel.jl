module AppCompIntel

    using Plots
    using StatsPlots
    using DataFrames

    export
        #Utils - Plots
        figure_path,
        get_category,

    include("utils.jl")

end # module
