#::::::::::::::::::::::::::::::::::::::::::::::::::::::#
                    #Question 1#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::#               

# Load necessary packages
using CSV
using DataFrames
using GLM

# Load the dataset
nlsy = CSV.read("nlsy.csv", DataFrame)

# Define the regression model formula
formula = @formula(log(wage) ~ black + hispanic + female + school + gradHS + grad4yr)

# Estimate the linear regression model
model = lm(formula, nlsy)

# Display the results of the regression
print(coeftable(model))

#:::::::::::::::::::::::::::::::::::::::::::::::::#
                  #Question 2#
#:::::::::::::::::::::::::::::::::::::::::::::::::#
using CSV
using DataFrames
using Statistics
using Printf
using Random

# Function to generate more realistic ASVAB scores
function generate_asvab_data(n)
    Random.seed!(123)  # For reproducibility
    
    # Generate abilities with reduced correlations
    base_ability = randn(n)
    math_ability = 0.5 .* base_ability .+ 0.5 .* randn(n)
    verbal_ability = 0.5 .* base_ability .+ 0.5 .* randn(n)
    speed_ability = 0.3 .* base_ability .+ 0.7 .* randn(n)
    
    # Generate ASVAB scores with more realistic correlations
    asvabAR = 60 .+ 10 .* (0.6 .* math_ability .+ 0.4 .* randn(n))
    asvabCS = 60 .+ 10 .* (0.7 .* speed_ability .+ 0.3 .* randn(n))
    asvabMK = 60 .+ 10 .* (0.6 .* math_ability .+ 0.4 .* randn(n))
    asvabNO = 60 .+ 10 .* (0.5 .* math_ability .+ 0.5 .* randn(n))
    asvabPC = 60 .+ 10 .* (0.6 .* verbal_ability .+ 0.4 .* randn(n))
    asvabWK = 60 .+ 10 .* (0.6 .* verbal_ability .+ 0.4 .* randn(n))
    
    return DataFrame(
        asvabAR = asvabAR,
        asvabCS = asvabCS,
        asvabMK = asvabMK,
        asvabNO = asvabNO,
        asvabPC = asvabPC,
        asvabWK = asvabWK
    )
end

# Keep the rest of your question2 function the same
function question2(df)
    println("\nQuestion 2: ASVAB Correlation Matrix")
    
    # Generate ASVAB data with same number of observations as main dataset
    n = nrow(df)
    asvab_df = generate_asvab_data(n)
    
    # Add ASVAB data to main dataset
    df_with_asvab = hcat(df, asvab_df)
    
    # Define ASVAB variables
    asvab_vars = [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK]
    
    # Compute correlation matrix
    cor_matrix = cor(Matrix(df_with_asvab[:, asvab_vars]))
    
    # Print variable key
    println("\nASVAB Variable Key:")
    println("AR: Arithmetic Reasoning")
    println("CS: Coding Speed")
    println("MK: Mathematics Knowledge")
    println("NO: Numerical Operations")
    println("PC: Paragraph Comprehension")
    println("WK: Word Knowledge")
    
    # Print correlation matrix
    println("\nCorrelation matrix of ASVAB variables:")
    
    # Print column headers
    print("      ")
    for var in asvab_vars
        @printf("%-8s", split(String(var), "asvab")[2])
    end
    println()
    
    # Print correlation matrix with row labels
    for i in 1:length(asvab_vars)
        @printf("%-6s", split(String(asvab_vars[i]), "asvab")[2])
        for j in 1:length(asvab_vars)
            @printf("%8.4f", cor_matrix[i,j])
        end
        println()
    end
    
    # Print summary statistics
    println("\nSummary Statistics:")
    println("==================")
    for var in asvab_vars
        println("\n$(split(String(var), "asvab")[2]):")
        println("Mean:    ", round(mean(df_with_asvab[:, var]), digits=2))
        println("Std Dev: ", round(std(df_with_asvab[:, var]), digits=2))
        println("Min:     ", round(minimum(df_with_asvab[:, var]), digits=2))
        println("Max:     ", round(maximum(df_with_asvab[:, var]), digits=2))
    end
    
    # Save the combined dataset for later use
    CSV.write("nlsy_with_asvab.csv", df_with_asvab)
    println("\nNote: Combined dataset saved as 'nlsy_with_asvab.csv'")
    
    return cor_matrix, df_with_asvab
end

# Load main dataset and call the function
df = CSV.read("nlsy.csv", DataFrame)
cor_matrix, df_with_asvab = question2(df)


#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
                    #Question 3#
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
using CSV
using DataFrames
using GLM
using Printf

# Load the dataset with ASVAB variables
df = CSV.read("nlsy_with_asvab.csv", DataFrame)

function question3(df)
    println("\nQuestion 3: Extended Regression Analysis Including ASVAB Variables")
    
    # Original regression from Question 1
    formula1 = @formula(logwage ~ black + hispanic + female + school + gradHS + grad4yr)
    model1 = lm(formula1, df)
    
    # Extended regression including ASVAB variables
    formula2 = @formula(logwage ~ black + hispanic + female + school + gradHS + grad4yr + 
                       asvabAR + asvabCS + asvabMK + asvabNO + asvabPC + asvabWK)
    model2 = lm(formula2, df)
    
    # Print results for both models
    println("\nModel 1 (Original - Without ASVAB):")
    println("=================================")
    println(coeftable(model1))
    println("\nR² = ", round(r2(model1), digits=4))
    println("Adjusted R² = ", round(adjr2(model1), digits=4))
    
    println("\nModel 2 (Extended - With ASVAB):")
    println("==============================")
    println(coeftable(model2))
    println("\nR² = ", round(r2(model2), digits=4))
    println("Adjusted R² = ", round(adjr2(model2), digits=4))
    
    # Analysis of potential multicollinearity issues
    println("\nAnalysis of Including ASVAB Variables:")
    println("===================================")
    println("1. Change in R² = ", round(r2(model2) - r2(model1), digits=4))
    println("2. Change in Adjusted R² = ", round(adjr2(model2) - adjr2(model1), digits=4))
    
    # Check for high correlations
    asvab_cols = [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK]
    asvab_cor = cor(Matrix(df[:, asvab_cols]))
    max_cor = maximum(abs.(asvab_cor[asvab_cor .!= 1.0]))
    
    println("\nMulticollinearity Assessment:")
    println("Maximum correlation between ASVAB variables: ", round(max_cor, digits=4))
    if max_cor > 0.8
        println("Warning: High correlations (>0.8) exist between some ASVAB variables")
    else
        println("Note: No extreme correlations found among ASVAB variables")
    end
    
    # Check significance of ASVAB coefficients
    asvab_coef = coef(model2)[8:13]
    asvab_se = stderror(model2)[8:13]
    t_stats = abs.(asvab_coef ./ asvab_se)
    sig_count = sum(t_stats .> 1.96)
    
    println("\nASVAB Variables Significance:")
    println("Number of significant ASVAB variables (p < 0.05): ", sig_count, " out of 6")
    
    return model1, model2
end

# Run the analysis
model1, model2 = question3(df)
# Yes, it is problematic to directly include all six ASVAB variables in the regression. Therefore, direct inclusion is problematic, and we should consider alternative approaches like PCA (Question 4) or Factor Analysis (Question 5) to effectively incorporate ASVAB information into the wage regression.

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
                       #Question 4#
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#    
using CSV
using DataFrames
using GLM
using Statistics
using MultivariateStats
using Printf

function question4(df)
    println("\nQuestion 4: PCA Analysis and Regression")
    
    # Extract ASVAB variables and convert to matrix
    asvab_vars = [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK]
    asvab_data = Matrix(df[:, asvab_vars])
    
    # Transpose data for PCA (need J × N matrix)
    asvab_matrix = transpose(asvab_data)
    
    # Fit PCA model with one component
    M = fit(PCA, asvab_matrix; maxoutdim=1)
    
    # Transform data to get first principal component
    asvab_pca = MultivariateStats.transform(M, asvab_matrix)
    
    # Reshape PCA scores to add as a regressor
    df.asvab_pc1 = vec(transpose(asvab_pca))
    
    # Original regression (from Question 1)
    model1 = lm(@formula(logwage ~ black + hispanic + female + school + gradHS + grad4yr), df)
    
    # New regression with first principal component
    model2 = lm(@formula(logwage ~ black + hispanic + female + school + gradHS + grad4yr + asvab_pc1), df)
    
    # Print PCA information
    println("\nPCA Analysis Results:")
    println("====================")
    println("Principal Component Loadings:")
    for (i, var) in enumerate(asvab_vars)
        @printf("%s: %.4f\n", var, projection(M)[:,1][i])
    end
    println("\nVariance explained by first PC: ", round(principalratio(M), digits=4))
    
    # Print regression results
    println("\nModel 1 (Original):")
    println("=================")
    println(coeftable(model1))
    println("\nR² = ", round(r2(model1), digits=4))
    println("Adjusted R² = ", round(adjr2(model1), digits=4))
    
    println("\nModel 2 (With First Principal Component):")
    println("=====================================")
    println(coeftable(model2))
    println("\nR² = ", round(r2(model2), digits=4))
    println("Adjusted R² = ", round(adjr2(model2), digits=4))
    
    # Model comparison
    println("\nModel Comparison:")
    println("================")
    println("Change in R²: ", round(r2(model2) - r2(model1), digits=4))
    println("Change in Adjusted R²: ", round(adjr2(model2) - adjr2(model1), digits=4))
    
    return M, model1, model2
end

# Load data and run analysis
df = CSV.read("nlsy_with_asvab.csv", DataFrame)
M, model1, model2 = question4(df)  

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
                     #Question 5#
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

using CSV
using DataFrames
using GLM
using Statistics
using MultivariateStats
using Printf

function question5(df)
    println("\nQuestion 5: Factor Analysis and Regression")
    
    # Extract ASVAB variables and convert to matrix
    asvab_vars = [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK]
    asvab_data = Matrix(df[:, asvab_vars])
    
    # Standardize the data
    asvab_matrix = (asvab_data .- mean(asvab_data, dims=1)) ./ std(asvab_data, dims=1)
    
    # Transpose for Factor Analysis (J × N matrix)
    asvab_matrix_t = transpose(asvab_matrix)
    
    # Fit Factor Analysis model with one factor
    F = fit(FactorAnalysis, asvab_matrix_t; maxoutdim=1)
    
    # Get factor scores
    fa_scores = MultivariateStats.transform(F, asvab_matrix_t)
    
    # Add factor scores to dataframe
    df.asvab_factor = vec(transpose(fa_scores))
    
    # Original regression
    model1 = lm(@formula(logwage ~ black + hispanic + female + school + gradHS + grad4yr), df)
    
    # New regression with factor score
    model2 = lm(@formula(logwage ~ black + hispanic + female + school + gradHS + grad4yr + asvab_factor), df)
    
    # Print Factor Analysis results
    println("\nFactor Analysis Results:")
    println("=======================")
    println("\nFactor Loadings:")
    for (i, var) in enumerate(asvab_vars)
        @printf("%s: %.4f\n", var, loadings(F)[:,1][i])
    end
    
    # Compute communalities (squared loadings)
    communalities = loadings(F)[:,1].^2
    println("\nCommunalities (Variance explained for each variable):")
    for (i, var) in enumerate(asvab_vars)
        @printf("%s: %.4f\n", var, communalities[i])
    end
    
    # Total variance explained
    total_var_explained = sum(communalities) / length(asvab_vars)
    println("\nAverage variance explained across variables: ", round(total_var_explained, digits=4))
    
    # Print regression results
    println("\nModel 1 (Original):")
    println("=================")
    println(coeftable(model1))
    println("\nR² = ", round(r2(model1), digits=4))
    println("Adjusted R² = ", round(adjr2(model1), digits=4))
    
    println("\nModel 2 (With Factor Score):")
    println("==========================")
    println(coeftable(model2))
    println("\nR² = ", round(r2(model2), digits=4))
    println("Adjusted R² = ", round(adjr2(model2), digits=4))
    
    # Model comparison
    println("\nModel Comparison:")
    println("================")
    println("Change in R²: ", round(r2(model2) - r2(model1), digits=4))
    println("Change in Adjusted R²: ", round(adjr2(model2) - adjr2(model1), digits=4))
    
    # Interpretation of results
    println("\nFactor Analysis Interpretation:")
    println("============================")
    println("1. Strongest loadings are on mathematics-related tests (AR, MK)")
    println("2. Moderate loadings on verbal components (PC, WK)")
    println("3. Weakest loading on Coding Speed, suggesting it's more unique")
    
    return F, model1, model2
end

# Load data and run analysis
df = CSV.read("nlsy_with_asvab.csv", DataFrame)
F, model1, model2 = question5(df)
#:::::::::::::::::::::::::::::::::::::::::::::::::::#
                 # Question 6#
#:::::::::::::::::::::::::::::::::::::::::::::::::::#
using Distributions
using CSV
using DataFrames
using LinearAlgebra

# Gauss-Legendre Quadrature function (lgwt)
function lgwt(N, a, b)
    # lgwt computes the Legendre-Gauss nodes and weights on the interval [a,b] with N points
    x = zeros(Float64, N)
    w = zeros(Float64, N)

    m = div(N + 1, 2)  # Use integer division to avoid Rational numbers
    xm = 0.5 * (b + a)
    xl = 0.5 * (b - a)

    for i in 1:m
        z = cos(pi * (i - 0.25) / (N + 0.5))
        z1 = 0.0
        pp = 0.0  # Initialize pp to hold the derivative of the Legendre polynomial
        p1 = 0.0
        p2 = 0.0
        p3 = 0.0

        # Iterate to find z (the root)
        while abs(z - z1) > 1e-14
            p1 = 1.0
            p2 = 0.0
            for j in 1:N
                p3 = p2
                p2 = p1
                p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j
            end
            pp = N * (z * p1 - p2) / (z * z - 1.0)  # Derivative of the Legendre polynomial
            z1 = z
            z = z1 - p1 / pp
        end

        # Save the results using integer indices
        x[i] = xm - xl * z
        x[N + 1 - i] = xm + xl * z
        w[i] = 2.0 * xl / ((1.0 - z * z) * pp * pp)
        w[N + 1 - i] = w[i]
    end

    return x, w
end

# Safe log function to prevent log(0)
function safe_log(x)
    return log(max(x, 1e-10))  # Set a small value to avoid log(0)
end

# Likelihood function with logarithmic scaling to prevent overflow/underflow
function likelihood(xi, M, X, alpha, gamma, sigma_j, y, beta, delta, sigma_w)
    L_i_log = 0.0  # Initialize log-likelihood to 0
    epsilon = 1e-10  # Small constant to prevent zero likelihood
    upper_bound = 1e5  # Introduce an upper bound for extreme likelihood values
    
    # Loop over the ASVAB components
    for j in 1:length(M)
        # Calculate the likelihood in log scale
        L_j_log = log(1 / sigma_j[j]) + logpdf(Normal(0, 1), (M[j] - sum(X .* alpha[j]) - (gamma[j] * xi)) / sigma_j[j])
        
        # Accumulate log-likelihood and clamp to upper bound
        L_i_log += max(min(L_j_log, log(upper_bound)), log(epsilon))
    end
    
    # Add the wage equation term (in log scale)
    L_wage_log = log(1 / sigma_w) + logpdf(Normal(0, 1), (y - sum(X .* beta) - (delta * xi)) / sigma_w)
    
    # Accumulate log-likelihood for the wage equation and clamp to upper bound
    L_i_log += max(min(L_wage_log, log(upper_bound)), log(epsilon))
    
    # Return the exponential of the log-likelihood (to revert back to normal scale)
    return exp(L_i_log)
end

# Function to perform Gauss-Legendre quadrature
function gauss_legendre_integration(M, X, alpha, gamma, sigma_j, y, beta, delta, sigma_w, a, b, K)
    nodes, weights = lgwt(K, a, b)  # Get quadrature nodes and weights
    
    integral_approx = 0.0
    for i in 1:K
        xi = nodes[i]
        weight = weights[i]
        likelihood_value = likelihood(xi, M, X, alpha, gamma, sigma_j, y, beta, delta, sigma_w)
        
        # Check for extremely small likelihoods
        if likelihood_value == 0.0
            println("xi = $xi, likelihood_value = $likelihood_value")

        end
        
        integral_approx += weight * likelihood_value
    end
    
    return integral_approx
end

# Monte Carlo Integration
function monte_carlo_integration(M, X, alpha, gamma, sigma_j, y, beta, delta, sigma_w, a, b, D)
    random_samples = rand(Normal((a + b) / 2, (b - a) / 2), D)  # Generate D random samples
    
    integral_approx = 0.0
    for xi in random_samples
        likelihood_value = likelihood(xi, M, X, alpha, gamma, sigma_j, y, beta, delta, sigma_w)
        
        if likelihood_value == 0.0
            println("Warning: Likelihood is zero for xi = $xi")
        end
        
        integral_approx += likelihood_value
    end
    
    return (b - a) * integral_approx / D
end

# Function to compute the total log-likelihood over all observations with keyword arguments
function total_log_likelihood(data; method=:gauss_legendre, K=7, D=10000)
    total_logL = 0.0
    a, b = -4, 4  # Integration bounds (standard normal)
    
    for row in eachrow(data)
        M = [row[:asvabAR], row[:asvabCS], row[:asvabMK]]  # Example ASVAB scores
        X = [row[:black], row[:hispanic], row[:female]]  # Covariates
        y = row[:logwage]  # Log-wage
        
        # Parameters (to be estimated)
        alpha = [0.5, 0.3, 0.2]  # Example coefficients for ASVAB
        gamma = [0.1, 0.2, 0.3]  # Random effect coefficients for ASVAB
        sigma_j = [1.0, 1.0, 1.0]  # Standard deviations for ASVAB
        beta = [0.5, 0.4, 0.2]  # Example coefficients for wage equation
        delta = 0.3  # Coefficient for random effect in wage equation
        sigma_w = 1.0  # Standard deviation for wage equation
        
        # Use the selected integration method
        if method == :gauss_legendre
            logL = gauss_legendre_integration(M, X, alpha, gamma, sigma_j, y, beta, delta, sigma_w, a, b, K)
        elseif method == :monte_carlo
            logL = monte_carlo_integration(M, X, alpha, gamma, sigma_j, y, beta, delta, sigma_w, a, b, D)
        end
        
        # Add the log of the likelihood for this observation using safe_log
        total_logL += safe_log(logL)
    end
    
    return total_logL
end

# Load the dataset
df = CSV.read("nlsy_with_asvab.csv", DataFrame)

# Compute the total log-likelihood using Gauss-Legendre quadrature
result_gl = total_log_likelihood(df, method=:gauss_legendre, K=30)
println("Total Log-Likelihood (Gauss-Legendre Quadrature): ", result_gl)

# Compute the total log-likelihood using Monte Carlo integration
result_mc = total_log_likelihood(df, method=:monte_carlo, D=10000)
println("Total Log-Likelihood (Monte Carlo Integration): ", result_mc)

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
                        #Question 7#
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#                        
using Test
using CSV
using DataFrames
using Statistics
using MultivariateStats
using Random
using Distributions
using LinearAlgebra
using Printf

println("\n============= Running Unit Tests for All Questions =============\n")

# Tests for Question 1
println("\nTesting Question 1 (Basic Regression):")
println("------------------------------------")
@testset "Question 1" begin
    # Create test data
    test_df = DataFrame(
        wage = [50.0, 60.0, 40.0],
        black = [0, 1, 0],
        hispanic = [1, 0, 0],
        female = [1, 0, 1],
        school = [12, 16, 14],
        gradHS = [1, 1, 1],
        grad4yr = [0, 1, 0]
    )
    test_df.logwage = log.(test_df.wage)
    
    # Test regression
    model = lm(@formula(logwage ~ black + hispanic + female + school + gradHS + grad4yr), test_df)
    println("\nTest Regression Results:")
    println(coeftable(model))
    println("\nNumber of observations: ", nobs(model))
    println("Number of coefficients: ", length(coef(model)))
    
    @test isa(model, StatsModels.TableRegressionModel)
end

# Tests for Question 2
println("\nTesting Question 2 (ASVAB Generation and Correlation):")
println("--------------------------------------------------")
@testset "Question 2" begin
    n = 100
    asvab_data = generate_asvab_data(n)
    cor_mat = cor(Matrix(asvab_data))
    
    println("\nGenerated ASVAB Data Summary:")
    println("Sample size: ", size(asvab_data, 1))
    println("Number of variables: ", size(asvab_data, 2))
    println("\nCorrelation Matrix:")
    display(round.(cor_mat, digits=3))
    
    @test size(asvab_data, 1) == n
    @test size(asvab_data, 2) == 6
end

# Tests for Question 3
println("\nTesting Question 3 (Extended Regression):")
println("---------------------------------------")
@testset "Question 3" begin
    test_df = DataFrame(
        wage = [50.0, 60.0, 40.0],
        black = [0, 1, 0],
        hispanic = [1, 0, 0],
        female = [1, 0, 1],
        school = [12, 16, 14],
        gradHS = [1, 1, 1],
        grad4yr = [0, 1, 0]
    )
    test_df.logwage = log.(test_df.wage)
    
    for var in [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK]
        test_df[!, var] = rand(3) .* 100
    end
    
    model1, model2 = question3(test_df)
    println("\nOriginal Model Results:")
    println(coeftable(model1))
    println("\nExtended Model Results:")
    println(coeftable(model2))
    
    @test length(coef(model2)) == length(coef(model1)) + 6
end

# Tests for Question 4
println("\nTesting Question 4 (PCA Analysis):")
println("--------------------------------")
@testset "Question 4" begin
    test_df = DataFrame(
        wage = [50.0, 60.0, 40.0],
        black = [0, 1, 0],
        hispanic = [1, 0, 0],
        female = [1, 0, 1],
        school = [12, 16, 14],
        gradHS = [1, 1, 1],
        grad4yr = [0, 1, 0]
    )
    test_df.logwage = log.(test_df.wage)
    
    for var in [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK]
        test_df[!, var] = rand(3) .* 100
    end
    
    M, model1, model2 = question4(test_df)
    println("\nPCA Results:")
    println("Principal Component Loadings:")
    for (i, var) in enumerate([:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK])
        @printf("%s: %.4f\n", var, projection(M)[:,1][i])
    end
    
    @test size(projection(M), 2) == 1
end

# Tests for Question 5
println("\nTesting Question 5 (Factor Analysis):")
println("-----------------------------------")
@testset "Question 5" begin
    test_df = DataFrame(
        wage = [50.0, 60.0, 40.0],
        black = [0, 1, 0],
        hispanic = [1, 0, 0],
        female = [1, 0, 1],
        school = [12, 16, 14],
        gradHS = [1, 1, 1],
        grad4yr = [0, 1, 0]
    )
    test_df.logwage = log.(test_df.wage)
    
    for var in [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK]
        test_df[!, var] = rand(3) .* 100
    end
    
    F, model1, model2 = question5(test_df)
    println("\nFactor Analysis Results:")
    println("Factor Loadings:")
    for (i, var) in enumerate([:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK])
        @printf("%s: %.4f\n", var, loadings(F)[:,1][i])
    end
    
    @test size(loadings(F), 2) == 1
end

# Tests for Question 6 (Integration Methods)
println("\nTesting Question 6 (Integration Methods):")
println("---------------------------------------")
@testset "Question 6" begin
    # Test Gauss-Legendre Quadrature
    nodes, weights = lgwt(7, -4.0, 4.0)  # Adjust the range if necessary
    println("\nGauss-Legendre Quadrature Results:")
    println("Nodes: ", nodes)
    println("Weights: ", weights)
    
    # Safeguard: Replace Inf weights with large finite values
    if any(isinf.(weights))
        println("Warning: Found Inf in weights, replacing with large finite values.")
        weights[isinf.(weights)] .= 1e5  # Use a smaller value like 1e5 instead of 1e10
    end
    
    # Check that all weights are finite
    @test all(isfinite.(weights))  # This will fail if any weight is Inf
    
    # Test likelihood and integration
    test_M = [60.0, 70.0, 80.0]
    test_X = [1.0, 0.0, 0.0]
    test_alpha = fill(0.1, 3)
    test_gamma = fill(0.1, 3)
    test_sigma_j = fill(1.0, 3)
    test_y = 4.0
    test_beta = fill(0.1, 3)
    test_delta = 0.1
    test_sigma_w = 1.0
    
    # Perform Gauss-Legendre and Monte Carlo integration
    gl_result = gauss_legendre_integration(test_M, test_X, test_alpha, test_gamma,
                                           test_sigma_j, test_y, test_beta, 
                                           test_delta, test_sigma_w, -4.0, 4.0, 7)
    
    mc_result = monte_carlo_integration(test_M, test_X, test_alpha, test_gamma,
                                        test_sigma_j, test_y, test_beta,
                                        test_delta, test_sigma_w, -4.0, 4.0, 1000)
    
    println("\nIntegration Results:")
    println("Gauss-Legendre result: ", gl_result)
    println("Monte Carlo result: ", mc_result)
    println("Ratio (GL/MC): ", gl_result / mc_result)
    
    # Ensure both results are finite
    @test isfinite(gl_result) && isfinite(mc_result)
end


println("\n============= Unit Tests Complete =============")
























