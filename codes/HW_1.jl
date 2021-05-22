using DataFrames, CSV, Plots

Income_df_sg = CSV.File("../data/ddf--gapminder--systema_globalis/countries-etc-datapoints/ddf--datapoints--income_per_person_gdppercapita_ppp_inflation_adjusted--by--geo--time.csv") |> DataFrame;

WorkingHours_df_sg = CSV.File("../data/ddf--gapminder--systema_globalis/countries-etc-datapoints/ddf--datapoints--working_hours_per_week--by--geo--time.csv") |> DataFrame;

LiteracyRate_df_sg = CSV.File("../data/ddf--gapminder--systema_globalis/countries-etc-datapoints/ddf--datapoints--literacy_rate_adult_total_percent_of_people_ages_15_and_above--by--geo--time.csv") |> DataFrame;

EnergyProd_df_sg = CSV.File("../data/ddf--gapminder--systema_globalis/countries-etc-datapoints/ddf--datapoints--energy_production_total--by--geo--time.csv") |> DataFrame;

WaterWithdraw_df_sg = CSV.File("../data/ddf--gapminder--systema_globalis/countries-etc-datapoints/ddf--datapoints--water_withdrawal_cu_meters_per_person--by--geo--time.csv") |> DataFrame;

LifeExpectancy_df_sg = CSV.File("../data/ddf--gapminder--systema_globalis/countries-etc-datapoints/ddf--datapoints--life_expectancy_years--by--geo--time.csv") |> DataFrame;

CivilSocPart_df_sg = CSV.File("../data/ddf--gapminder--fasttrack/ddf--datapoints--c_civsocpart_idea--by--country--time.csv") |> DataFrame;

GDP_df_sg = CSV.File("../data/ddf--gapminder--systema_globalis/countries-etc-datapoints/ddf--datapoints--gdp_total_yearly_growth--by--geo--time.csv") |> DataFrame;

Gini_df_sg = CSV.File("../data/ddf--gapminder--systema_globalis/countries-etc-datapoints/ddf--datapoints--gapminder_gini--by--geo--time.csv") |> DataFrame;


