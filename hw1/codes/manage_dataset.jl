concrete_df = dropmissing(CSV.File("./data/Concrete_Data.csv") |> DataFrame)

strength_min = minimum(concrete_df[:, "Concrete compressive strength (MPa)"])
strength_max = maximum(concrete_df[:, "Concrete compressive strength (MPa)"])

function get_category(strength, strength_max, strength_min)
    strength_step =  (strength_max - strength_min) / 5
    if strength <= strength_step
        return 1
    elseif strength_step < strength <= 2 * strength_step
        return 2
    elseif  2 * strength_step < strength <= 3 * strength_step
        return 3
    elseif  3 * strength_step < strength <= 4 * strength_step
        return 4 
    elseif strength > 4 * strength_step
        return 5
    end
end

transform!(concrete_df, "Concrete compressive strength (MPa)" => ByRow(strength -> get_category(strength, strength_max, strength_min)) => "Category")

open("./data/ConcreteUCI.csv", "w") do file
    CSV.write(file, concrete_df)
end
