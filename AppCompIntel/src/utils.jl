function get_category(strength)
    if strength < 25 # Non-standard -> https://www.baseconcrete.co.uk/different-types-of-concrete-grades-and-their-uses/
        return 1
    elseif 25 <= strength < 50 # Standard
        return 2
    elseif  50 <= strength # High Strength
        return 3
    end
end

function figure_path(figure_name)
    return eval(@__DIR__)*"/../../hw1/figures/"*figure_name
end