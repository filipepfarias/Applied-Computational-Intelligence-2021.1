module AppCompIntel

    using Plots
    using StatsPlots
    using DataFrames

    export
        #Utils - Plots
        plot_monovariate_histograms, plot_scatters, plot_bivariate_scatters

    include("utils.jl")

end # module
