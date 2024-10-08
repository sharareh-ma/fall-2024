using CSV, DataFrames, DataFramesMeta
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
                          #Question 1#
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#                         
# Specify the file path
file_path = "/Users/sanazma/fall-2024-10/ProblemSets/PS5-ddc/busdata.csv"

println("Attempting to use file: ", file_path)

if isfile(file_path)
    # Read in the data
    df = CSV.read(file_path, DataFrame)

    # Display the column names
    println("Columns in the file: ", names(df))

    # Identify Y columns and Odo columns
    y_cols = filter(x -> startswith(string(x), "Y"), names(df))
    odo_cols = filter(x -> startswith(string(x), "Odo"), names(df))

    # Create a long format dataframe
    df_long = DataFrame(
        Bus = repeat(1:nrow(df), inner=length(y_cols)),
        Time = repeat(1:length(y_cols), outer=nrow(df)),
        Y = vcat([df[!, col] for col in y_cols]...),
        Odometer = vcat([df[!, col] for col in odo_cols]...),
        RouteUsage = repeat(df.RouteUsage, inner=length(y_cols)),
        Branded = repeat(df.Branded, inner=length(y_cols))
    )

    # Sort the dataframe
    sort!(df_long, [:Bus, :Time])

    # Display the first few rows of the reshaped data
    println("\nFirst few rows of the reshaped data:")
    println(first(df_long, 5))

    # Display summary statistics
    println("\nSummary statistics:")
    describe(df_long)

    # Save the long format data
    CSV.write("busdata_long.csv", df_long)
    println("\nLong format data saved as 'busdata_long.csv'")
else
    println("File not found: ", file_path)
    println("Please ensure the data file is in the correct location.")
end

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
                      #Question 2#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
using CSV, DataFrames, GLM

# Load the long-format data we created in Question 1
df_long = CSV.read("busdata_long.csv", DataFrame)

# Create additional variables for the model
df_long.Mileage2 = df_long.Odometer.^2
df_long.RouteUsage2 = df_long.RouteUsage.^2
df_long.Time2 = df_long.Time.^2

# Create the formula using @formula macro
formula = @formula(Y ~ (Odometer + Mileage2 + RouteUsage + RouteUsage2 + Branded + Time + Time2)^2)

# Estimate the logit model
model = glm(formula, df_long, Binomial(), LogitLink())

# Display the model summary
println(model)

# Save the results to a file
open("logit_results.txt", "w") do io
    println(io, model)
end

println("\nModel results have been saved to 'logit_results.txt'")

# Get predicted probabilities
df_long.predicted_prob = predict(model)

# Display the first few rows of the dataframe with predicted probabilities
println("\nFirst few rows with predicted probabilities:")
println(first(df_long[:, [:Bus, :Time, :Y, :Odometer, :RouteUsage, :Branded, :predicted_prob]], 5))

# Save the updated dataframe
CSV.write("busdata_long_with_predictions.csv", df_long)
println("\nUpdated data with predictions saved as 'busdata_long_with_predictions.csv'")

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
                        #Question 3#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#    

using CSV, DataFrames, GLM, LinearAlgebra, Statistics, Random

# Load the long-format data we created in Question 1
df_long = CSV.read("busdata_long_with_predictions.csv", DataFrame)

# Define the discount factor
β = 0.9

# Part (a): Construct the state transition matrices
function construct_transition_matrices(df)
    # Get unique values for states
    xvals = sort(unique(df.Odometer))
    zvals = sort(unique(df.RouteUsage))
    xbin = length(xvals)
    zbin = length(zvals)

    # Initialize transition matrices
    xtran = zeros(xbin * zbin, xbin)
    
    # Compute transitions
    for t in 1:size(df, 1)-1
        if df.Bus[t] == df.Bus[t+1]  # Ensure we're looking at the same bus
            x1 = findfirst(==(df.Odometer[t]), xvals)
            x2 = findfirst(==(df.Odometer[t+1]), xvals)
            z = findfirst(==(df.RouteUsage[t]), zvals)
            row = (z - 1) * xbin + x1
            xtran[row, x2] += 1
        end
    end

    # Normalize transitions
    for i in 1:size(xtran, 1)
        s = sum(xtran[i, :])
        if s > 0
            xtran[i, :] ./= s
        end
    end

    return xtran, xvals, zvals, xbin, zbin
end

xtran, xvals, zvals, xbin, zbin = construct_transition_matrices(df_long)

println("Part (a) completed: State transition matrices constructed")

# Part (b): Compute future value terms
function compute_future_values(df, xtran, xvals, zvals, xbin, zbin, β)
    T = maximum(df.Time)
    B = unique(df.Branded)
    
    # Create a dataframe for all possible states
    state_df = DataFrame(
        Odometer = repeat(xvals, outer=zbin),
        RouteUsage = repeat(zvals, inner=xbin),
        Branded = zeros(Int, xbin*zbin),
        Time = zeros(Int, xbin*zbin)
    )

    # Initialize future value array
    FV = zeros(xbin * zbin, length(B), T + 1)

    # Refit the logit model to get CCPs
    ccp_formula = @formula(Y ~ Odometer + RouteUsage + Branded + Time)
    ccp_model = glm(ccp_formula, df, Binomial(), LogitLink())

    for t in T:-1:2
        for b in B
            state_df.Time .= t
            state_df.Branded .= b
            p0 = 1 .- predict(ccp_model, state_df)
            FV[:, Int(b)+1, t] = -β * log.(p0)
        end
    end

    return FV
end

FV = compute_future_values(df_long, xtran, xvals, zvals, xbin, zbin, β)

println("Part (b) completed: Future value terms computed")

# Part (c): Estimate structural parameters
function estimate_structural_parameters(df, xtran, FV, xvals, zvals, xbin, zbin, β)
    println("Dimensions of FV: ", size(FV))
    println("xbin: ", xbin)
    println("zbin: ", zbin)
    println("Max Time in data: ", maximum(df.Time))

    # Prepare data for estimation
    X = [ones(nrow(df)) df.Odometer df.Branded]
    
    # Compute the future value term for each observation
    fv_term = zeros(nrow(df))
    for i in 1:nrow(df)
        x1 = findfirst(==(df.Odometer[i]), xvals)
        z = findfirst(==(df.RouteUsage[i]), zvals)
        row = (z - 1) * xbin + x1
        b = Int(df.Branded[i]) + 1
        t = df.Time[i]
        
        if t < maximum(df.Time)
            end_row = min(row+xbin-1, size(FV, 1))
            fv_diff = FV[row:end_row, b, t+1] - xtran[row, 1:length(row:end_row)] .* FV[row, b, t+1]
            fv_term[i] = sum(xtran[row, 1:length(row:end_row)] .* fv_diff)
        end
    end
    
    # Add the future value term to X
    X = hcat(X, fv_term)
    
    # Estimate parameters using logistic regression
    y = df.Y
    
    # Add a small constant to avoid log(0)
    epsilon = 1e-10
    y_adj = y .* (1 - 2*epsilon) .+ epsilon
    
    # Use try-catch to handle potential errors
    try
        θ = (X' * X) \ (X' * log.(y_adj ./ (1 .- y_adj)))
        return θ, X, fv_term
    catch e
        println("Error in parameter estimation: ", e)
        return [NaN, NaN, NaN, NaN], X, fv_term
    end
end

θ, X, fv_term = estimate_structural_parameters(df_long, xtran, FV, xvals, zvals, xbin, zbin, β)

println("Part (c) completed: Structural parameters estimated")
println("Estimated parameters: ", θ)

# Save results
open("structural_results.txt", "w") do io
    println(io, "Structural Parameter Estimates:")
    println(io, "θ₀ (Constant): ", θ[1])
    println(io, "θ₁ (Odometer): ", θ[2])
    println(io, "θ₂ (Branded): ", θ[3])
    println(io, "θ₃ (Future Value): ", θ[4])
end

println("Results saved to 'structural_results.txt'")

# Additional diagnostics
println("\nAdditional Diagnostics:")
println("Range of y values: ", extrema(df_long.Y))
println("Range of Odometer values: ", extrema(df_long.Odometer))
println("Range of Branded values: ", extrema(df_long.Branded))
println("Range of fv_term values: ", extrema(fv_term))
println("Condition number of X'X: ", cond(X'X))

# Calculate and print the mean of the future value term
println("Mean of future value term: ", mean(fv_term))

# Calculate and print correlations
println("\nCorrelations:")
println("Correlation between Odometer and fv_term: ", cor(df_long.Odometer, fv_term))
println("Correlation between Branded and fv_term: ", cor(df_long.Branded, fv_term))

# Print summary statistics of X
println("\nSummary of X:")
for i in 1:size(X, 2)
    println("Column $i: mean = $(mean(X[:, i])), std = $(std(X[:, i]))")
end

# Part (d): Custom logit function with restricted offset term
function custom_logit_with_offset(X, y, offset)
    function log_likelihood(β)
        z = X * β .+ offset
        p = 1 ./ (1 .+ exp.(-z))
        return -sum(y .* log.(p .+ 1e-10) + (1 .- y) .* log.(1 .- p .+ 1e-10))
    end
    
    result = optimize(log_likelihood, zeros(size(X, 2)))
    return Optim.minimizer(result)
end

# Part (e): Wrap all code in an empty function and time it
function wrapper_function()
    # Construct transition matrices
    xtran, xvals, zvals, xbin, zbin = construct_transition_matrices(df_long)

    # Compute future values
    FV = compute_future_values(df_long, xtran, xvals, zvals, xbin, zbin, β)

    # Calculate future value term
    fv_term = calculate_fv_term(df_long, FV, xtran, xvals, zvals, xbin, zbin)

    # Prepare data for estimation
    X = Matrix(df_long[:, [:Odometer, :Branded]])
    y = df_long.Y

    # Estimate the model using custom logit function with offset
    β_hat = custom_logit_with_offset(X, y, fv_term)

    return β_hat
end

# Time the execution
@time results = try
    wrapper_function()
catch e
    println("An error occurred: ", e)
    nothing
end

if !isnothing(results)
    println("Estimated parameters:")
    println("β₁ (Odometer): ", results[1])
    println("β₂ (Branded): ", results[2])
else
    println("Estimation failed. Please check the error message above.")
end

# Part (f): Glory in the power of CCPs!
println("\nDiscussion on the power of CCPs:")
println("Conditional Choice Probabilities (CCPs) offer several advantages in dynamic discrete choice models:")
println("1. Computational Efficiency: CCPs allow us to avoid solving the full dynamic programming problem, significantly reducing computational time.")
println("2. Flexibility: They enable the use of flexible specifications for the choice probabilities without requiring parametric assumptions about the distribution of unobservables.")
println("3. Two-Step Estimation: CCPs facilitate a two-step estimation process, where we first estimate choice probabilities and then use these to estimate structural parameters.")
println("4. Handling Large State Spaces: CCPs are particularly useful when dealing with models that have large state spaces, where traditional methods might be computationally infeasible.")
println("5. Reduced-Form Insights: The first-stage CCP estimates can provide useful reduced-form insights about choice behavior.")
println("\nIn our bus engine replacement model, CCPs allow us to estimate the structural parameters without repeatedly solving the dynamic programming problem, making the estimation process more tractable and efficient.")

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
                             #Questin 4#
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#                             
import StatsModels
using Test, CSV, DataFrames, GLM, Random, Statistics

println("Starting the bus engine replacement model tests...")

# Helper function to create a sample dataset for testing
function create_sample_data()
    Random.seed!(123)
    n = 100
    df = DataFrame(
        Bus = repeat(1:20, inner=5),
        Time = repeat(1:5, outer=20),
        Odometer = rand(100000:200000, n),
        RouteUsage = rand(1:5, n),
        Branded = rand(0:1, n),
        Y = rand(0:1, n)
    )
    println("Sample data created with $(nrow(df)) rows and $(ncol(df)) columns.")
    return df
end

# Question 1: Data Reshaping
function reshape_to_long(df)
    long_df = stack(df, [:Odometer, :RouteUsage, :Branded], [:Bus, :Time])
    rename!(long_df, :variable => :Measure, :value => :Value)
    println("Data reshaped to long format. New shape: $(size(long_df))")
    return long_df
end

# Question 2: Flexible Logit Model
function estimate_flexible_logit(df)
    formula = StatsModels.@formula(Y ~ Odometer * Odometer^2 * RouteUsage * RouteUsage^2 * Branded * Time * Time^2)
    model = glm(formula, df, Binomial(), LogitLink())
    println("Flexible logit model estimated. Number of coefficients: $(length(coef(model)))")
    return model
end

# Question 3: CCP Estimation (Placeholder functions)
function construct_transition_matrices(df)
    println("Constructing transition matrices (placeholder)")
    return zeros(100, 100), zeros(100, 100)
end

function compute_future_values(df, flex_logit_model, xtran, zval, xbin, T)
    println("Computing future values (placeholder)")
    return zeros(size(df, 1))
end

function estimate_structural_parameters(df_long, fv)
    df_long.fv = fv
    formula = StatsModels.@formula(Y ~ Odometer + Branded + fv)
    model = glm(formula, df_long, Binomial(), LogitLink())
    println("Structural parameters estimated. Number of coefficients: $(length(coef(model)))")
    return model
end

# Test functions
@testset "Bus Engine Replacement Model Tests" begin
    println("\nStarting tests...")

    # Tests for Question 1
    @testset "Data Reshaping (Question 1)" begin
        println("\nTesting data reshaping...")
        df = create_sample_data()
        df_long = reshape_to_long(df)
        
        @test size(df_long, 1) == 3 * size(df, 1)
        @test "Measure" in names(df_long)
        @test "Value" in names(df_long)
        @test Set(names(df_long)) == Set(["Bus", "Time", "Measure", "Value"])
        println("Data reshaping tests completed.")
    end

    # Tests for Question 2
    @testset "Flexible Logit Model (Question 2)" begin
        println("\nTesting flexible logit model...")
        df = create_sample_data()
        model = estimate_flexible_logit(df)
        
        @test isa(model, StatsModels.TableRegressionModel)
        @test isa(model.mf.f.lhs, StatsModels.ContinuousTerm)
        @test model.mf.f.lhs.sym == :Y
        @test length(coef(model)) > 7
        println("Flexible logit model tests completed.")
    end

    # Tests for Question 3
    @testset "CCP Estimation (Question 3)" begin
        println("\nTesting CCP estimation...")
        df = create_sample_data()
        
        flex_logit_model = estimate_flexible_logit(df)
        xtran, ztran = construct_transition_matrices(df)
        fv = compute_future_values(df, flex_logit_model, xtran, df.RouteUsage, 100, 5)
        structural_model = estimate_structural_parameters(df, fv)

        @test isa(xtran, Matrix)
        @test isa(ztran, Matrix)
        @test size(fv, 1) == size(df, 1)
        @test isa(structural_model, StatsModels.TableRegressionModel)
        @test length(coef(structural_model)) == 4  # Intercept + 3 variables
        println("CCP estimation tests completed.")
    end
end

println("\nAll tests completed!")