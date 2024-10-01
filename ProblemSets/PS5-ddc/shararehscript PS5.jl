using DataFrames, CSV, HTTP, DataFramesMeta
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
                     #Question#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
function reshape_bus_data()
    # Load the data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)

    # Create a bus_id if it doesn't exist
    if !(:bus_id in names(df))
        df.bus_id = 1:nrow(df)
    end

    # Identify the columns to reshape
    y_cols = [Symbol("Y$i") for i in 1:20]
    odo_cols = [Symbol("Odo$i") for i in 1:20]

    # Reshape Y columns
    df_y = stack(df, y_cols, [:bus_id, :RouteUsage, :Branded], variable_name=:time, value_name=:decision)

    # Reshape Odometer columns
    df_odo = stack(df, odo_cols, [:bus_id], variable_name=:time, value_name=:odometer)

    # Clean up the time variable
    df_y.time = parse.(Int, replace.(string.(df_y.time), r"[^0-9]" => ""))
    df_odo.time = parse.(Int, replace.(string.(df_odo.time), r"[^0-9]" => ""))

    # Merge the reshaped dataframes
    df_long = innerjoin(df_y, df_odo, on=[:bus_id, :time])

    # Sort the dataframe
    sort!(df_long, [:bus_id, :time])

    return df_long
end

# Reshape the data
df_long = reshape_bus_data()

# Display the first few rows of the reshaped data
println(first(df_long, 10))

#:::::::::::::::::::::::::::::::::::::::::::::#
                #Question2#
#:::::::::::::::::::::::::::::::::::::::::::::#
using GLM, DataFrames, CSV, HTTP, DataFramesMeta

# First, let's make sure we have the reshaped data from Question 1
function reshape_bus_data()
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    
    if !(:bus_id in names(df))
        df.bus_id = 1:nrow(df)
    end
    
    y_cols = [Symbol("Y$i") for i in 1:20]
    odo_cols = [Symbol("Odo$i") for i in 1:20]
    
    df_y = stack(df, y_cols, [:bus_id, :RouteUsage, :Branded], variable_name=:time, value_name=:decision)
    df_odo = stack(df, odo_cols, [:bus_id], variable_name=:time, value_name=:odometer)
    
    df_y.time = parse.(Int, replace.(string.(df_y.time), r"[^0-9]" => ""))
    df_odo.time = parse.(Int, replace.(string.(df_odo.time), r"[^0-9]" => ""))
    
    df_long = innerjoin(df_y, df_odo, on=[:bus_id, :time])
    sort!(df_long, [:bus_id, :time])
    
    return df_long
end

df_long = reshape_bus_data()

# Now, let's estimate the binary logit model
# The flow utility of running (not replacing) a bus is:
# u_1(x_1t, b) = θ_0 + θ_1 * x_1t + θ_2 * b
# where x_1t is the mileage (odometer reading) and b is the branded dummy

# Estimate the logit model
logit_model = glm(@formula(decision ~ odometer + Branded), df_long, Binomial(), LogitLink())

# Print the summary of the model
println(logit_model)

# Extract the coefficients
θ_0 = coef(logit_model)[1]  # Intercept
θ_1 = coef(logit_model)[2]  # Coefficient for odometer
θ_2 = coef(logit_model)[3]  # Coefficient for Branded

println("Estimated coefficients:")
println("θ_0 (Intercept): ", θ_0)
println("θ_1 (Odometer): ", θ_1)
println("θ_2 (Branded): ", θ_2)

#:::::::::::::::::::::::::::::::::::::::::::::::::::::#
                    #Question 3#
#:::::::::::::::::::::::::::::::::::::::::::::::::::::#
using DataFrames, CSV, HTTP, DataFramesMeta, Optim, LinearAlgebra

# Include the create_grids function
include("create_grids.jl")

# Define a struct to hold all the data and parameters
struct BusData
    N::Int
    T::Int
    β::Float64
    X::Matrix{Float64}
    Y::Matrix{Int}
    B::Vector{Int}
    Xstate::Matrix{Int}
    Zstate::Vector{Int}
    zbin::Int
    xbin::Int
    xval::Vector{Float64}
    xtran::Matrix{Float64}
end

function estimate_dynamic_model()
    # Part (a): Read in the data for the dynamic model
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS5-ddc/busdata.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)

    Y = Array(df[:, r"^Y"])
    Odo = Array(df[:, r"^Odo"])
    Xst = Array(df[:, r"^Xst"])
    Zst = df.Zst

    # Part (b): Construct the state transition matrices
    zval, zbin, xval, xbin, xtran = create_grids()

    # Create the BusData struct
    data_parms = BusData(
        size(Y, 1),  # N
        size(Y, 2),  # T
        0.9,         # β
        Odo,         # X
        Y,           # Y
        df.Branded,  # B
        Xst,         # Xstate
        Zst,         # Zstate
        zbin,
        xbin,
        xval,
        xtran
    )

    # Part (e): Wrap everything into a function for Optim
    @views @inbounds function likebus(θ, d)
        # Part (c): Compute future value terms for all possible states of the model
        FV = zeros(d.zbin*d.xbin, 2, d.T+1)
        
        for t = d.T:-1:1
            for b = 0:1
                for z = 1:d.zbin
                    for x = 1:d.xbin
                        row = x + (z-1)*d.xbin
                        v1 = θ[1] + θ[2]*d.xval[x] + θ[3]*b + d.β * d.xtran[row,:]' * FV[(z-1)*d.xbin+1:z*d.xbin,b+1,t+1]
                        v0 = d.β * d.xtran[1+(z-1)*d.xbin,:]' * FV[(z-1)*d.xbin+1:z*d.xbin,b+1,t+1]
                        
                        FV[row,b+1,t] = d.β * log(exp(v1) + exp(v0))
                    end
                end
            end
        end

        # Part (d): Construct the log likelihood
        like = 0.0
        for i = 1:d.N
            row0 = (d.Zstate[i]-1)*d.xbin+1
            for t = 1:d.T
                row1 = d.Xstate[i,t] + (d.Zstate[i]-1)*d.xbin
                v1 = θ[1] + θ[2]*d.X[i,t] + θ[3]*d.B[i] + d.β * (d.xtran[row1,:].-d.xtran[row0,:])' * FV[row0:row0+d.xbin-1,d.B[i]+1,t+1]
                dem = 1 + exp(v1)
                like -= ((d.Y[i,t]==1)*v1) - log(dem)
            end
        end
        return like
    end

    # Part (f): Optimize using Optim
    θ_start = rand(3)
    θ̂_optim = optimize(θ -> likebus(θ, data_parms), θ_start, LBFGS(), 
                       Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true))
    
    θ̂_ddc = Optim.minimizer(θ̂_optim)

    # Print results
    println("\nOptimization Results:")
    println("Converged: ", Optim.converged(θ̂_optim))
    println("Minimum: ", Optim.minimum(θ̂_optim))
    println("\nEstimated Parameters:")
    println("θ₁ (Constant): ", θ̂_ddc[1])
    println("θ₂ (Odometer): ", θ̂_ddc[2])
    println("θ₃ (Branded): ", θ̂_ddc[3])

    return θ̂_ddc
end
#g wrap all
using CSV, DataFrames, HTTP, Optim, LinearAlgebra

# Load the dataset
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS5-ddc/busdata.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Extract the matrices and vectors
Y = Matrix(df[:, r"Y[0-9]+"])     # Extract Y matrix (decision matrix)
Odo = Matrix(df[:, r"Odo[0-9]+"]) # Extract Odo matrix (odometer readings)
B = df.Branded                    # Extract B vector (branding status)
Xst = Matrix(df[:, r"Xst[0-9]+"]) # Extract Xst matrix (state variable for transitions)
Zst = df.Zst                      # Extract Zst vector (state variable for usage intensity)

# Create the grids and transition matrix
zval, zbin, xval, xbin, xtran = create_grids()  # Assuming the function is defined

# Define the compute_future_values function
function compute_future_values(θ, β)
    T = 20  # Assuming the time horizon is 20 periods, adjust this as per your data
    FV = zeros(xbin * zbin, 2, T + 1)  # 3D array: states × decision (replace or not) × time

    # Loop backward through time, from period T to 1
    for t in T:-1:1
        for b in 0:1  # b = 0 means not replaced, b = 1 means replaced
            for z in 1:zbin  # Loop over route usage intensity bins
                for x in 1:xbin  # Loop over odometer states
                    row = x + (z-1)*xbin

                    # Flow utility of not replacing the bus
                    v1 = θ[1] + θ[2]*xval[x] + θ[3]*b + β * dot(xtran[row, :], FV[(z-1)*xbin+1:z*xbin, b+1, t+1])

                    # Flow utility of replacing the bus (resetting the odometer)
                    v0 = β * dot(xtran[1 + (z-1)*xbin, :], FV[(z-1)*xbin+1:z*xbin, b+1, t+1])

                    # Store the future value for this state and decision
                    FV[row, b+1, t] = β * log(exp(v0) + exp(v1))
                end
            end
        end
    end

    return FV
end

# Wrapping the entire process into a function called "myfun"
@views @inbounds function wrap_myfun(θ::Vector{Float64}, N::Int, T::Int, Odo::Matrix{Float64}, Y::Matrix{Int}, B::Vector{Int}, Xst::Matrix{Int}, Zst::Vector{Int}, xtran::Matrix{Float64}, xbin::Int)
    # Compute future values for all possible states
    FV = compute_future_values(θ, 0.9)

    # Initialize log-likelihood
    ll = 0.0

    # Loop over buses and time periods to compute the log-likelihood
    for i in 1:N
        for t in 1:T
            # Index for the case where the bus has been replaced (mileage reset to 0)
            row0 = 1 + (Zst[i] - 1) * xbin
            
            # Index for the case where the bus has not been replaced (mileage accumulated)
            row1 = Xst[i,t] + (Zst[i] - 1) * xbin

            # Flow utility component (flow utility for not replacing the bus)
            v1 = θ[1] + θ[2] * Odo[i,t] + θ[3] * B[i]

            # Add the discounted future value difference
            v1 += 0.9 * (xtran[row1,:] .- xtran[row0,:])' * FV[row0:row0+xbin-1, B[i]+1, t+1]

            # Compute choice probabilities
            P1 = exp(v1) / (1 + exp(v1))  # Probability of not replacing

            # Update log-likelihood
            ll += Y[i,t] * log(P1) + (1 - Y[i,t]) * log(1 - P1)
        end
    end

    # Return the negative log-likelihood since we're maximizing the likelihood
    return -ll
end

# Example of how to use wrap_myfun in the optimization process (Section g)
function section_g_optimization()
    # Load or define the necessary data: N, T, Odo, Y, B, Xst, Zst, xtran, xbin
    N = size(Y, 1)  # Number of buses
    T = size(Y, 2)  # Number of time periods
    θ_start = [0.0, 0.0, 0.0]  # Initial guess for parameters θ₁, θ₂, θ₃

    # Perform optimization using Optim.jl
    result = optimize(θ -> wrap_myfun(θ, N, T, Odo, Y, B, Xst, Zst, xtran, xbin), θ_start, LBFGS(), Optim.Options(g_tol = 1e-5, iterations = 100_000, show_trace = true))

    # Extract the optimized parameters
    θ_optimized = Optim.minimizer(result)
    
    # Print results
    println("Optimization Results:")
    println("Converged: ", Optim.converged(result))
    println("Minimum value of the objective function (Log-Likelihood): ", Optim.minimum(result))
    println("Estimated Parameters: ")
    println("θ₁ (Constant): ", θ_optimized[1])
    println("θ₂ (Odometer): ", θ_optimized[2])
    println("θ₃ (Branded): ", θ_optimized[3])

    return θ_optimized
end

# Call the optimization function
section_g_optimization()

#g
using Optim, DataFrames, CSV, HTTP

# Helper function to load data
function load_data()
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdata.csv"
    return CSV.read(HTTP.get(url).body, DataFrame)
end

# Define the dynamic model optimization function
function run_dynamic_model_optimization()
    # Load the data using the helper function
    df = load_data()

    # Extract the relevant matrices and vectors
    Y = Matrix(df[:, r"Y[0-9]+"])     # Decision matrix (Y)
    Odo = Matrix(df[:, r"Odo[0-9]+"]) # Odometer readings (Odo)
    B = df.Branded                    # Branded status (B)
    Xst = Matrix(df[:, r"Xst[0-9]+"]) # State variable transitions (Xst)
    Zst = df.Zst                      # Usage intensity state (Zst)

    # Create the grids and transition matrix (assuming the create_grids function is already defined)
    zval, zbin, xval, xbin, xtran = create_grids()

    # Set initial parameter guesses for θ₁, θ₂, θ₃
    θ_initial = [0.0, 0.0, 0.0]  # Starting with zero initial values for parameters

    # Perform optimization using the wrap_myfun function (assuming wrap_myfun is defined)
    result = optimize(θ -> wrap_myfun(θ, size(Y, 1), size(Y, 2), Odo, Y, B, Xst, Zst, xtran, xbin),
                      θ_initial, LBFGS(), Optim.Options(g_tol = 1e-5, iterations = 100_000, show_trace = true))

    # Extract the optimized parameters
    θ_optimized = Optim.minimizer(result)

    # Return the optimized parameters and log-likelihood value
    return θ_optimized, Optim.minimum(result)
end

# Step 2: Run the optimization and print the results
θ_optimized, log_likelihood_value = run_dynamic_model_optimization()

println("Optimization Results:")
println("Log-Likelihood Value: ", log_likelihood_value)
println("θ₁ (Constant): ", θ_optimized[1])
println("θ₂ (Odometer Effect): ", θ_optimized[2])
println("θ₃ (Branded Effect): ", θ_optimized[3])

# Interpretation of results
function interpret_parameters(θ_optimized)
    # Extract individual parameters
    θ1 = θ_optimized[1]
    θ2 = θ_optimized[2]
    θ3 = θ_optimized[3]

    # Print Interpretation
    println("\n--- Interpretation of Parameters ---")
    
    # Intercept (θ₁)
    println("θ₁ (Constant):")
    println("A baseline effect representing the log-odds of not replacing a bus when the odometer is 0 and the bus is not branded.")
    println("Value: $θ1")
    
    # Odometer Effect (θ₂)
    println("\nθ₂ (Odometer Effect):")
    if θ2 < 0
        println("Negative coefficient suggests that buses with higher mileage are more likely to be replaced.")
    else
        println("Unexpected positive coefficient: Higher mileage is associated with a lower likelihood of replacement.")
    end
    println("Value: $θ2")

    # Branded Effect (θ₃)
    println("\nθ₃ (Branded Effect):")
    if θ3 > 0
        println("Positive coefficient suggests that branded buses are more likely to be replaced.")
    else
        println("Unexpected negative coefficient: Branded buses are less likely to be replaced.")
    end
    println("Value: $θ3")

    # Log-Likelihood interpretation
    println("\n--- Model Fit ---")
    println("The log-likelihood value is: $log_likelihood_value. This provides a measure of how well the model fits the data.")
end

# Run the interpretation function
interpret_parameters(θ_optimized)

#::::::::::::::::::::::::::::::::::::::::::::::::::::#
                   #Question4#
#::::::::::::::::::::::::::::::::::::::::::::::::::::#
using Test, DataFrames, CSV, HTTP, GLM, Optim, LinearAlgebra, Random, Statistics

# Helper function to load data
function load_data()
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdata.csv"
    return CSV.read(HTTP.get(url).body, DataFrame)
end

# Function to reshape bus data
function reshape_bus_data()
    # Load the data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdata.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)

    # Create a bus_id if it doesn't exist
    if !(:bus_id in names(df))
        df.bus_id = 1:nrow(df)
    end

    # Identify the columns to reshape
    y_cols = [Symbol("Y$i") for i in 1:20]
    odo_cols = [Symbol("Odo$i") for i in 1:20]

    # Reshape Y columns (decision)
    df_y = stack(df, y_cols, [:bus_id, :RouteUsage, :Branded], variable_name=:time, value_name=:decision)

    # Reshape Odometer columns
    df_odo = stack(df, odo_cols, [:bus_id], variable_name=:time, value_name=:odometer)

    # Clean up the time variable
    df_y.time = parse.(Int, replace.(string.(df_y.time), r"[^0-9]" => ""))
    df_odo.time = parse.(Int, replace.(string.(df_odo.time), r"[^0-9]" => ""))

    # Merge the reshaped dataframes
    df_long = innerjoin(df_y, df_odo, on=[:bus_id, :time])

    # Sort the dataframe
    sort!(df_long, [:bus_id, :time])

    # Debugging: Print column names to verify they are correct
    println("Columns in df_long: ", names(df_long))

    return df_long
end
@testset "Question 1: Reshape Data" begin
    df_long = reshape_bus_data()
    @test isa(df_long, DataFrame)
    @test nrow(df_long) == nrow(load_data()) * 20

    # Convert Symbols to Strings for comparison
    expected_columns = ["bus_id", "time", "decision", "odometer", "Branded"]
    @test all(col in names(df_long) for col in expected_columns)
end

# Test for Question 2: Static Logit Model
@testset "Question 2: Static Logit Model" begin
    df_long = reshape_bus_data()
    model = glm(@formula(decision ~ odometer + Branded), df_long, Binomial(), LogitLink())

    # Check that the model is a TableRegressionModel wrapping a GeneralizedLinearModel
    @test isa(model, StatsModels.TableRegressionModel)
    @test isa(model.model, GLM.GeneralizedLinearModel)  # Check the internal model type

    # Ensure that the coefficients are finite
    @test all(isfinite, coef(model))
    
    # Test that the coefficient for odometer is negative (as expected)
    @test coef(model)[2] < 0
end
println("Tests for Question 2 completed.")

# Test for Question 3: Dynamic Model
# Function to create grids for state variables
function create_grids()
    # Example: grid values for z (say, marketing conditions) and x (say, odometer states)
    zval = range(0.0, 1.0, length=5)  # A grid for some state variable z
    xval = range(0.0, 100.0, length=10)  # A grid for odometer or some other state variable x

    zbin = length(zval)  # Number of bins for z
    xbin = length(xval)  # Number of bins for x

    # Transition matrix for x (e.g., odometer)
    # This is a dummy transition matrix; you should replace it with actual transition dynamics
    xtran = Matrix{Float64}(I, xbin, xbin)

    return zval, zbin, xval, xbin, xtran
end
@testset "Question 3: Dynamic Model" begin
    df = load_data()
    Y = Matrix(df[:, r"Y[0-9]+"])
    Odo = Matrix(df[:, r"Odo[0-9]+"])
    Xst = Matrix(df[:, r"Xst[0-9]+"])
    Zst = df.Zst
    B = df.Branded

    # Test create_grids function
    zval, zbin, xval, xbin, xtran = create_grids()
    @test length(zval) == zbin
    @test length(xval) == xbin
    @test size(xtran) == (xbin, xbin)

    # Additional dynamic model tests here
end
println("Tests for Question 3 completed.")

println("All unit tests completed successfully!")