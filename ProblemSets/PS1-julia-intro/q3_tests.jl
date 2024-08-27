# q3.jl

using CSV
using DataFrames

function q3()
    # Reading in data and processing
    nlsw88 = CSV.read("nlsw88.csv", DataFrame)
    
    # Handling missing values (if needed)
    # Assuming missing values are already handled during reading

    # Calculating summary statistics
    summarystats = describe(nlsw88)

    # Frequency tables
    race_table = freqtable(nlsw88, :race)

    # Cross-tabulation of industry and occupation
    industry_occupation_crosstab = combine(groupby(nlsw88, [:industry, :occupation]), :wage => mean => :mean_wage)

    # Return all results for further use
    return nlsw88, summarystats, race_table, industry_occupation_crosstab
end

