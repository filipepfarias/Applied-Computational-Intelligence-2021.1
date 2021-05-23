using DataFrames, CSV, Plots

Income_df_sg = CSV.File("./data/ddf--gapminder--systema_globalis/countries-etc-datapoints/ddf--datapoints--income_per_person_gdppercapita_ppp_inflation_adjusted--by--geo--time.csv") |> DataFrame;

WorkingHours_df_sg = CSV.File("./data/ddf--gapminder--systema_globalis/countries-etc-datapoints/ddf--datapoints--working_hours_per_week--by--geo--time.csv") |> DataFrame;

LiteracyRate_df_sg = CSV.File("./data/ddf--gapminder--systema_globalis/countries-etc-datapoints/ddf--datapoints--literacy_rate_adult_total_percent_of_people_ages_15_and_above--by--geo--time.csv") |> DataFrame;

EnergyProd_df_sg = CSV.File("./data/ddf--gapminder--systema_globalis/countries-etc-datapoints/ddf--datapoints--energy_production_total--by--geo--time.csv") |> DataFrame;

WaterWithdraw_df_sg = CSV.File("./data/ddf--gapminder--systema_globalis/countries-etc-datapoints/ddf--datapoints--water_withdrawal_cu_meters_per_person--by--geo--time.csv") |> DataFrame;

LifeExpectancy_df_sg = CSV.File("./data/ddf--gapminder--systema_globalis/countries-etc-datapoints/ddf--datapoints--life_expectancy_years--by--geo--time.csv") |> DataFrame;

CivilSocPart_df_sg = CSV.File("./data/ddf--gapminder--fasttrack/ddf--datapoints--c_civsocpart_idea--by--country--time.csv") |> DataFrame;

GDP_df_sg = CSV.File("./data/ddf--gapminder--systema_globalis/countries-etc-datapoints/ddf--datapoints--gdp_total_yearly_growth--by--geo--time.csv") |> DataFrame;

Gini_df_sg = CSV.File("./data/ddf--gapminder--systema_globalis/countries-etc-datapoints/ddf--datapoints--inequality_index_gini--by--geo--time.csv") |> DataFrame;

function GetBrazilData(df_var)
    if any(names(df_var) .== "geo")
        return select!(filter(row -> row.geo == "bra",df_var),Not(:geo))
    elseif any(names(df_var) .== "country")
        return select!(filter(row -> row.country == "bra",df_var),Not(:country))
    else
        error("DataFrame column does not match!")
    end
end

HW1_df = outerjoin(([Income_df_sg,WorkingHours_df_sg,LiteracyRate_df_sg,EnergyProd_df_sg,WaterWithdraw_df_sg,LifeExpectancy_df_sg,CivilSocPart_df_sg,GDP_df_sg,Gini_df_sg] .|> GetBrazilData)...,on = :time)

# BRA_Gini = Gini_df_sg[Gini_df_sg[!,:geo].=="bra",:inequality_index_gini]
# BRA_Year = Gini_df_sg[Gini_df_sg[!,:geo].=="bra",:time]

# scatter(BRA_Year,BRA_Gini)Lite