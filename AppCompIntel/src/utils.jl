function get_category(strength)
    if strength < 25 # Non-standard -> https://www.baseconcrete.co.uk/different-types-of-concrete-grades-and-their-uses/
        return "L1"
    elseif 25 <= strength < 50 # Standard
        return "L2"
    elseif  50 <= strength # High Strength
        return "L3"
    end
end

function figure_path(figure_name)
    return eval(@__DIR__)*"/../hw2/figures/"*figure_name
end

function scatterplot(data::AbstractDataFrame; kw...)
    dims       = names(data) |> length
    fig_matrix = Matrix{}(undef,dims,dims);
    I          = LinearIndices(fig_matrix);
    cor_matrix = cor(Matrix(data));

    for i in 1:size(fig_matrix)[1]
        for j in 1:size(fig_matrix)[2]
            if i == j
                fig_matrix[i,i] = @df data groupedhist(cols(i), axis=true, ticks=false, legend=false, bins=10, lw = 0.1, framestyle = :box, kw...)
            elseif i>j
                fig_matrix[i,j] = @df data scatter(cols(i), cols(j), axis=true, ticks=false, legend=false; markerstrokecolor=:white, markerstrokewidth=0.0, markersize=2.7, framestyle = :box, kw...)
            else
                fig_matrix[i,j] = heatmap([1.0],[1.0],hcat([cor_matrix[i,j]]),
                    clim = (-1,1), c=:diverging_bwr_20_95_c54_n256, colorbar = false, axis=false, ticks = false)
                annotate!(
                    (1.,1.,(string(round(cor_matrix[i,j],digits=2)),12))
                    )
            end
            
            j == 1 ? plot!(fig_matrix[I[i,j]], xguide=L"D_{%$i}",xmirror = false, xguideposition= :bottom) : plot!(fig_matrix[I[i,j]], bottom_margin=-2Plots.mm)
            i == 1 ? plot!(fig_matrix[I[i,j]], yguide=L"D_{%$j}", ymirror = true, yguideposition= :left) : plot!(fig_matrix[I[i,j]], left_margin=-2Plots.mm)
        end
    end
    h2 = scatter([0,0], [0,1], zcolor=[0,3], clim=(-1,1),
                 xlims=(1,1.1), label="", c=:diverging_bwr_20_95_c54_n256, framestyle=:none)
    l = @layout [grid(dims,dims) a{0.035w}]

    fig_matrix = reverse(fig_matrix, dims=2);
    figure = plot(fig_matrix...,h2,layout = l, dpi=90, size=(710,650));

    return figure
end