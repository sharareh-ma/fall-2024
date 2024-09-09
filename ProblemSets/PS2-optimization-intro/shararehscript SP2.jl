#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
                          #Quetion 1#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#                         
                          
using Optim

# Define the function f(x)
f(x) = -x[1]^4 - 10x[1]^3 - 2x[1]^2 - 3x[1] - 2

# Define the negative of the function for minimization
negf(x) = x[1]^4 + 10x[1]^3 + 2x[1]^2 + 3x[1] + 2

# Start with a random starting value
startval = rand(1)

# Perform the optimization
result = optimize(negf, startval, LBFGS())

# Print the result
println("The maximum value of f(x) occurs at x = ", Optim.minimizer(result))
println("The maximum value of f(x) is ", -Optim.minimum(result))




#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
                      #Question 2#
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::#


using DataFrames
using CSV
using HTTP
using Optim
using LinearAlgebra

# Step 1: Load the data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Step 2: Prepare the data
# Create the independent variable matrix X (including intercept, age, race, college grad)
X = [ones(size(df,1)) df.age df.race .== 1 df.collgrad .== 1]

# Dependent variable (whether married or not)
y = df.married .== 1

# Step 3: Define the OLS objective function (SSR)
function ols_objective(beta, X, y)
    residuals = y .- X * beta
    ssr = residuals' * residuals  # sum of squared residuals
    return ssr
end

# Step 4: Estimate OLS using the Optim package
initial_beta = rand(size(X, 2))  # Random initial guess for coefficients
result = optimize(b -> ols_objective(b, X, y), initial_beta, LBFGS())

# Print the estimated coefficients
println("Estimated coefficients using Optim: ", Optim.minimizer(result))

# Step 5: Manual OLS using matrix algebra for comparison
beta_manual = inv(X' * X) * (X' * y)
println("Estimated coefficients using matrix algebra: ", beta_manual)




#::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
                      #Question 3#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::# 

using Optim
using LinearAlgebra
using Random
using DataFrames
using CSV
using HTTP

# Step 1: Load the data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Step 2: Prepare the data (assume 2 choices for simplicity)
X = [ones(size(df, 1)) df.age df.race .== 1 df.collgrad .== 1]
y = df.married .== 1  # Binary outcome (1 if married, 0 otherwise)

# Step 3: Define the logit likelihood function
function logit_likelihood(beta, X, y)
    # Compute linear predictor X * beta
    u = X * beta
    # Compute probabilities using the logit formula
    P = exp.(u) ./ (1 .+ exp.(u))
    # Log-likelihood (y is 1 or 0, so we use the binary logit form)
    ll = sum(y .* log.(P) .+ (1 .- y) .* log.(1 .- P))
    return -ll  # Return the negative log-likelihood (since we are minimizing)
end

# Step 4: Optimize the negative log-likelihood
initial_beta = rand(size(X, 2))  # Initial guess for coefficients
result = optimize(b -> logit_likelihood(b, X, y), initial_beta, LBFGS())

# Print the estimated coefficients
println("Estimated coefficients using Optim: ", Optim.minimizer(result))

# Step 5: Compare with GLM
using GLM
logit_model = glm(@formula(married ~ age + race + collgrad), df, Binomial(), LogitLink())
println("Estimated coefficients using GLM: ", coef(logit_model))




#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
                    #Question 4#
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
using Optim
using DataFrames
using CSV
using HTTP
using GLM

# Step 1: Load the data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Step 2: Prepare the data (binary outcome)
X = [ones(size(df, 1)) df.age df.race .== 1 df.collgrad .== 1]
y = df.married .== 1  # Binary dependent variable

# Step 3: Define the logit log-likelihood function
function logit_likelihood(beta, X, y)
    # Calculate linear predictor
    z = X * beta
    # Calculate probabilities P using the logit function
    P = 1 ./ (1 .+ exp.(-z))
    # Calculate log-likelihood
    ll = sum(y .* log.(P) .+ (1 .- y) .* log.(1 .- P))
    return -ll  # Return negative log-likelihood (since we minimize)
end

# Step 4: Use Optim to maximize the log-likelihood (minimize the negative)
initial_beta = rand(size(X, 2))  # Initial guess for coefficients
result = optimize(b -> logit_likelihood(b, X, y), initial_beta, LBFGS())

# Step 5: Print estimated coefficients
println("Estimated coefficients using Optim: ", Optim.minimizer(result))

# Step 6: Compare with GLM
logit_model = glm(@formula(married ~ age + race + collgrad), df, Binomial(), LogitLink())
println("Estimated coefficients using GLM: ", coef(logit_model))




#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
                          #Question 5#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#                          

using DataFrames
using CSV
using HTTP
using FreqTables
using Optim

# Step 1: Load the data from the URL
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Step 2: Check the frequency table of occupation to see small categories
println("Frequency of Occupation before aggregation:")
display(freqtable(df, :occupation))

# Remove rows where occupation is missing
df = dropmissing(df, :occupation)

# Step 3: Aggregate small categories (8, 9, 10, 11, 12, 13 into category 7)
df[df.occupation .== 8, :occupation] .= 7
df[df.occupation .== 9, :occupation] .= 7
df[df.occupation .== 10, :occupation] .= 7
df[df.occupation .== 11, :occupation] .= 7
df[df.occupation .== 12, :occupation] .= 7
df[df.occupation .== 13, :occupation] .= 7

# Verify that the problem is solved
println("Frequency of Occupation after aggregation:")
display(freqtable(df, :occupation))

# Step 4: Define X and y
X = [ones(size(df,1)) df.age df.race .== 1 df.collgrad .== 1]  # Independent variables
y = df.occupation  # Dependent variable (occupation)

# Step 5: Define the softmax (multinomial logit) probability function
function softmax(X, beta)
    exps = exp.(X * beta)  # Exponentiate X * beta for each category
    return exps ./ sum(exps, dims=2)  # Normalize to get probabilities
end

# Step 6: Define the negative log-likelihood for multinomial logit
function multinomial_loglikelihood_neg(beta, X, y, K)
    N = size(X, 1)  # Number of observations
    beta = reshape(beta, (size(X, 2), K))  # Reshape beta into a matrix with K columns
    probs = softmax(X, beta)  # Compute the probabilities
    ll = 0.0
    for i in 1:N
        ll += log(probs[i, y[i]])  # Log-likelihood for each observation
    end
    return -ll  # Return negative log-likelihood
end

# Step 7: Number of categories (K) after aggregating
K = 7

# Step 8: Starting values for beta (randomized)
beta_init = randn(size(X, 2) * K)

# Step 9: Perform optimization using L-BFGS algorithm
result = optimize(b -> multinomial_loglikelihood_neg(b, X, y, K), beta_init, LBFGS(), Optim.Options(g_tol=1e-5))

# Step 10: Print the estimated coefficients (minimizer)
beta_hat = reshape(result.minimizer, (size(X, 2), K))
println("Estimated coefficients (multinomial logit model): ")
display(beta_hat)





#::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
                    #Question 6#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
# ::::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 5: Multinomial Logit Model for Occupational Choice
# ::::::::::::::::::::::::::::::::::::::::::::::::::::

using DataFrames
using CSV
using HTTP
using FreqTables
using Optim
using LinearAlgebra

# Step 1: Load the data from the URL
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Step 2: Check the frequency table of occupation to see small categories
println("Frequency of Occupation before aggregation:")
display(freqtable(df, :occupation))

# Remove rows where occupation is missing
df = dropmissing(df, :occupation)

# Step 3: Aggregate small categories (8, 9, 10, 11, 12, 13 into category 7)
df[df.occupation .== 8, :occupation] .= 7
df[df.occupation .== 9, :occupation] .= 7
df[df.occupation .== 10, :occupation] .= 7
df[df.occupation .== 11, :occupation] .= 7
df[df.occupation .== 12, :occupation] .= 7
df[df.occupation .== 13, :occupation] .= 7

# Verify that the problem is solved
println("Frequency of Occupation after aggregation:")
display(freqtable(df, :occupation))

# Step 4: Define X and y
X = [ones(size(df,1)) df.age df.race .== 1 df.collgrad .== 1]  # Independent variables
y = df.occupation  # Dependent variable (occupation)

# Step 5: Define the softmax (multinomial logit) probability function
function softmax(X, beta)
    exps = exp.(X * beta)  # Exponentiate X * beta for each category
    return exps ./ sum(exps, dims=2)  # Normalize to get probabilities
end

# Step 6: Define the negative log-likelihood for multinomial logit
function multinomial_loglikelihood_neg(beta, X, y, K)
    N = size(X, 1)  # Number of observations
    beta = reshape(beta, (size(X, 2), K))  # Reshape beta into a matrix with K columns
    probs = softmax(X, beta)  # Compute the probabilities
    ll = 0.0
    for i in 1:N
        ll += log(probs[i, y[i]])  # Log-likelihood for each observation
    end
    return -ll  # Return negative log-likelihood
end

# Step 7: Number of categories (K) after aggregating
K = 7

# Step 8: Starting values for beta (randomized)
beta_init = randn(size(X, 2) * K)

# Step 9: Perform optimization using L-BFGS algorithm
result = optimize(b -> multinomial_loglikelihood_neg(b, X, y, K), beta_init, LBFGS(), Optim.Options(g_tol=1e-5))

# Step 10: Print the estimated coefficients (minimizer)
beta_hat = reshape(result.minimizer, (size(X, 2), K))
println("Estimated coefficients (multinomial logit model):")
display(beta_hat)

# ::::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 6: Model Evaluation and Interpretation
# ::::::::::::::::::::::::::::::::::::::::::::::::::::

using Statistics

# 1. Interpretation of Coefficients
println("Estimated coefficients (multinomial logit model):")
display(beta_hat)

# Interpretation:
println("\nInterpretation of coefficients:")
println("For each occupational category, the estimated coefficients reflect the log-odds of selecting that occupation, compared to the baseline category.")
println("A positive coefficient indicates an increase in the likelihood of that occupation with an increase in the corresponding variable (e.g., age, race, college graduation).")

# 2. Predicted Probabilities for Sample Individuals

# Define the sample individuals as 1x4 matrices (row vectors for each individual)
individual1 = [1.0 30.0 1.0 1.0]  # 30 years old, white, college graduate
individual2 = [1.0 50.0 0.0 0.0]  # 50 years old, non-white, non-college graduate

# Predict probabilities using the softmax function and estimated beta_hat
predicted_probs1 = softmax(individual1, beta_hat)
predicted_probs2 = softmax(individual2, beta_hat)

println("\nPredicted probabilities for individual 1 (30, white, college graduate):")
display(predicted_probs1)

println("Predicted probabilities for individual 2 (50, non-white, non-college graduate):")
display(predicted_probs2)

# 3. Model Goodness of Fit: AIC and BIC

# Extract the log-likelihood from the optimization result
log_likelihood = -result.minimum  # Optim returns the negative log-likelihood

# Calculate AIC and BIC
n = size(X, 1)  # Number of observations
k = length(result.minimizer)  # Number of estimated parameters

aic = 2k - 2 * log_likelihood
bic = k * log(n) - 2 * log_likelihood

println("\nModel Fit Statistics:")
println("Log-Likelihood: ", log_likelihood)
println("AIC: ", aic)
println("BIC: ", bic)

# 4. Model Improvements (Discussion in code comments)

println("\nDiscussion on Model Improvements:")
println("""
Potential model improvements include:
1. Adding more covariates such as work experience, marital status, or geographic region to better capture individual characteristics that influence occupational choice.
2. Addressing potential limitations of the multinomial logit model, such as the assumption of Independence of Irrelevant Alternatives (IIA). 
   To address this, consider using alternative models like nested logit or mixed logit models that can relax the IIA assumption.
3. Assessing interactions between covariates (e.g., age and education) might improve the model's predictive power.
4. Evaluating the model's out-of-sample predictive performance using cross-validation or hold-out samples.
""")




#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
                           #Question 7
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
# ::::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 5: Multinomial Logit Model for Occupational Choice
# ::::::::::::::::::::::::::::::::::::::::::::::::::::

using DataFrames
using CSV
using HTTP
using FreqTables
using Optim
using LinearAlgebra

# Step 1: Load the data from the URL
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Step 2: Check the frequency table of occupation to see small categories
println("Frequency of Occupation before aggregation:")
display(freqtable(df, :occupation))

# Remove rows where occupation is missing
df = dropmissing(df, :occupation)

# Step 3: Aggregate small categories (8, 9, 10, 11, 12, 13 into category 7)
df[df.occupation .== 8, :occupation] .= 7
df[df.occupation .== 9, :occupation] .= 7
df[df.occupation .== 10, :occupation] .= 7
df[df.occupation .== 11, :occupation] .= 7
df[df.occupation .== 12, :occupation] .= 7
df[df.occupation .== 13, :occupation] .= 7

# Verify that the problem is solved
println("Frequency of Occupation after aggregation:")
display(freqtable(df, :occupation))

# Step 4: Define X and y
X = [ones(size(df,1)) df.age df.race .== 1 df.collgrad .== 1]  # Independent variables
y = df.occupation  # Dependent variable (occupation)

# Step 5: Define the softmax (multinomial logit) probability function
function softmax(X, beta)
    exps = exp.(X * beta)  # Exponentiate X * beta for each category
    return exps ./ sum(exps, dims=2)  # Normalize to get probabilities
end

# Step 6: Define the negative log-likelihood for multinomial logit
function multinomial_loglikelihood_neg(beta, X, y, K)
    N = size(X, 1)  # Number of observations
    beta = reshape(beta, (size(X, 2), K))  # Reshape beta into a matrix with K columns
    probs = softmax(X, beta)  # Compute the probabilities
    ll = 0.0
    for i in 1:N
        ll += log(probs[i, y[i]])  # Log-likelihood for each observation
    end
    return -ll  # Return negative log-likelihood
end

# Step 7: Number of categories (K) after aggregating
K = 7

# Step 8: Starting values for beta (randomized)
beta_init = randn(size(X, 2) * K)

# Step 9: Perform optimization using L-BFGS algorithm
result = optimize(b -> multinomial_loglikelihood_neg(b, X, y, K), beta_init, LBFGS(), Optim.Options(g_tol=1e-5))

# Step 10: Print the estimated coefficients (minimizer)
beta_hat = reshape(result.minimizer, (size(X, 2), K))
println("Estimated coefficients (multinomial logit model):")
display(beta_hat)

# ::::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 7: Predicting New Observations and Calculating Accuracy
# ::::::::::::::::::::::::::::::::::::::::::::::::::::

# Step 1: Define the softmax function (if not already defined)
# (Already defined in Question 5, so we don't need to redefine it)

# Step 2: Define a set of new individuals for prediction (features: intercept, age, race, college)
new_individuals = [
    [1.0 25.0 1.0 1.0];  # 25 years old, white, college graduate
    [1.0 40.0 0.0 0.0];  # 40 years old, non-white, non-college graduate
    [1.0 30.0 1.0 0.0];  # 30 years old, white, non-college graduate
    [1.0 55.0 0.0 1.0];  # 55 years old, non-white, college graduate
]

# Step 3: Predict probabilities for the new individuals using the softmax function and estimated beta_hat
predicted_probs_new = softmax(new_individuals, beta_hat)

# Display predicted probabilities for the new individuals
println("\nPredicted probabilities for new individuals:")
display(predicted_probs_new)

# Step 4: Assign the predicted occupation based on the highest probability for each individual
predicted_occupations = map(eachrow(predicted_probs_new)) do row
    argmax(row)  # Find the index (occupation category) with the highest probability
end

# Display the predicted occupational categories
println("\nPredicted occupational categories for new individuals:")
display(predicted_occupations)

# Step 5: Assume the true occupational categories for these new individuals
true_occupations = [3, 2, 1, 5]  # Hypothetical true categories (for example purposes)

# Step 6: Calculate the classification accuracy
correct_predictions = sum(predicted_occupations .== true_occupations)
accuracy = correct_predictions / length(true_occupations)

println("\nClassification accuracy: ", accuracy)













                    

