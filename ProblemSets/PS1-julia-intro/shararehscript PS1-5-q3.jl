using CSV
using DataFrames
using FreqTables
using Statistics  # Add this line to use the mean function
using Test

# Function from Question 3
function q3()
    nlsw88 = CSV.read("/Users/sanazma/fall-2024-9/ProblemSets/PS1-julia-intro/nlsw88.csv", DataFrame)
    
    # Process and analyze the data
    summarystats = describe(nlsw88)
    race_table = freqtable(nlsw88, :race)
    industry_occupation_crosstab = combine(groupby(nlsw88, [:industry, :occupation]), :wage => mean => :mean_wage)
    
    return nlsw88, summarystats, race_table, industry_occupation_crosstab
end

# Run the function to get the results
nlsw88, summarystats, race_table, industry_occupation_crosstab = q3()

# Unit Tests for the function
@testset "q3 Tests" begin
    @test !isempty(nlsw88)
    @test !isempty(summarystats)
    @test !isempty(race_table)
    @test !isempty(industry_occupation_crosstab)
end



