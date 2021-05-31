module AppCompIntel

    using Plots
    using StatsPlots
    using DataFrames

    export
        #Utils - Plots
        plot_monovariate_histograms, 
        plot_bivariate_scatters,
        figure_path,
        save_if_isfile,
        get_category,
        boxcox_transform,
        yeojohnson_transform

    include("utils.jl")

end # module
