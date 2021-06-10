module AppCompIntel

    using Plots
    using StatsPlots
    using DataFrames
    using YeoJohnsonTrans
    
    export
        #Utils - Plots
        plot_monovariate_histograms, 
        plot_bivariate_scatters,
        figure_path,
        save_if_isfile,
        get_category

    include("utils.jl")

end # module
