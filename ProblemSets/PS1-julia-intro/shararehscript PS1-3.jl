# Import necessary packages
import Pkg

# Ensure required packages are installed
try
    using StatsBase, DataFrames, CSV, Statistics
catch
    Pkg.add("StatsBase")
    Pkg.add("DataFrames")
    Pkg.add("CSV")
    Pkg.add("Statistics")
    using StatsBase, DataFrames, CSV, Statistics
end

# Define the function q3()
function q3()
    # (a) Import the file nlsw88.csv into Julia as a DataFrame
    file_path = "/Users/sanazma/fall-2024-4/ProblemSets/PS1-julia-intro/nlsw88.csv"
    
    if !isfile(file_path)
        println("ERROR: The file $file_path does not exist.")
        return
    end

    df = CSV.File(file_path, missingstring="NA") |> DataFrame
    
    # Print the column names to identify the correct ones
    println("Column names: ", names(df))
    
    # Save the processed DataFrame as nlsw88_processed.csv
    CSV.write("nlsw88_processed.csv", df)
    println("Data imported and saved as nlsw88_processed.csv.")
    
    # (b) Calculate the percentage of the sample that has never been married
    never_married_pct = sum(df[!, :married] .== 0) / nrow(df) * 100
    println("Percentage never married: ", never_married_pct, "%")
    
    # Replace :college with the correct column name after inspecting the output
    college_column_name = :college  # Update this with the correct column name after inspection
    if college_column_name in names(df)
        college_graduate_pct = sum(df[!, college_column_name] .== 1) / nrow(df) * 100
        println("Percentage college graduates: ", college_graduate_pct, "%")
    else
        println("The column :college was not found in the DataFrame.")
    end  # This ends the if block
    
    # (c) Alternative method to calculate the percentage of the sample in each race category
    race_counts = combine(groupby(df, :race), nrow => :count)
    race_counts[!, :percentage] = race_counts[!, :count] ./ sum(race_counts[!, :count]) * 100
    println("Percentage in each race category: ", race_counts)
    
    # (d) Use the describe() function to create summary statistics
    summarystats = describe(df)
    println("Summary statistics: ")
    println(summarystats)
    
    # How many grade observations are missing?
    missing_grades = sum(ismissing.(df[!, :grade]))
    println("Number of missing grade observations: ", missing_grades)
    
    # (e) Show the joint distribution of industry and occupation using a cross-tabulation
    industry_occupation_crosstab = combine(groupby(df, [:industry, :occupation]), nrow => :count)
    println("Joint distribution of industry and occupation (cross-tabulation): ")
    println(industry_occupation_crosstab)
    
    # (f) Tabulate the mean wage over industry and occupation categories
    df_subset = select(df, [:industry, :occupation, :wage])
    mean_wage = combine(groupby(df_subset, [:industry, :occupation]), :wage => mean => :mean_wage)
    println("Mean wage over industry and occupation categories: ")
    
    return nothing
end  # This ends the function q3

# Call the function q3() at the end of the script
q3()

