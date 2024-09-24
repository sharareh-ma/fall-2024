#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
                                        #Questin 1#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#                                        

using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables, Distributions, ForwardDiff

# Load data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS4-mixture/nlsw88t.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)
X = [df.age df.white df.collgrad]
Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
         df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
y = df.occ_code

# Define multinomial logit log-likelihood function
function mnl_loglikelihood(params, X, Z, y)
    N, K = size(X)
    J = size(Z, 2)
    
    β = reshape(params[1:K*(J-1)], K, J-1)
    γ = params[end]
    
    ll = 0.0
    for i in 1:N
        utilities = [X[i,:] ⋅ β[:,j] + γ * (Z[i,j] - Z[i,end]) for j in 1:(J-1)]
        push!(utilities, 0.0)  # Reference category
        probabilities = exp.(utilities) ./ sum(exp.(utilities))
        ll += log(probabilities[y[i]])
    end
    
    return -ll  # Return negative log-likelihood for minimization
end

# Estimation function
function estimate_mnl(X, Z, y)
    N, K = size(X)
    J = size(Z, 2)
    
    # Use the start values from PS3
    start_values = [0.05570767876416688, 0.08342649976722213, -2.344887681361976, 0.04500076157943125, 0.7365771540890512, -3.153244238810631, 0.09264606406280998, -0.08417701777996893, -4.273280002738097, 0.023903455659102114, 0.7230648923377259, -3.749393470343111, 0.03608733246865346, -0.6437658344513095, -4.2796847340030375, 0.0853109465190059, -1.1714299392376775, -6.678677013966667, 0.086620198654063, -0.7978777029320784, -4.969132023685069, -0.0941942241795243]
    
    result = optimize(params -> mnl_loglikelihood(params, X, Z, y),
                      start_values,
                      BFGS(),
                      Optim.Options(show_trace = true, iterations = 100000);
                      autodiff = :forward)
    
    estimates = Optim.minimizer(result)
    se = sqrt.(diag(inv(ForwardDiff.hessian(params -> mnl_loglikelihood(params, X, Z, y), estimates))))
    
    return estimates, se
end

# Run estimation
estimates, standard_errors = estimate_mnl(X, Z, y)

# Print results
println("Multinomial Logit Estimates with Alternative-Specific Covariates:")
for i in 1:length(estimates)
    println("Parameter $i: ", round(estimates[i], digits=6), " (", round(standard_errors[i], digits=6), ")")
end


#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
                                        #Question 2#            
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#                                      
# Extract and print γ estimate
gamma_hat = estimates[end]
gamma_se = standard_errors[end]
println("\nEstimated γ: ", round(gamma_hat, digits=6), " (", round(gamma_se, digits=6), ")")

# Calculate and print t-statistic for γ
t_stat = gamma_hat / gamma_se
println("t-statistic for γ: ", round(t_stat, digits=4))


# γ: 1.307477 
#The estimated coefficient γ̂ from Problem Set 4 (1.307477) makes more sense than the estimate from Problem Set 3 (-0.0943) for several reasons:
#Sign: The positive sign in Problem Set 4 aligns better with economic theory. A positive γ suggests that higher wages increase the probability of choosing an occupation, which is intuitively correct. The negative sign in Problem Set 3 was counterintuitive.
#Magnitude: The larger magnitude in Problem Set 4 suggests a stronger effect of wages on occupational choice. This is more realistic given the importance of wages in career decisions.
  

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
                                      #Question 3#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

# a. Practice using quadrature

using Distributions
include("lgwt.jl")

# Define distribution
d = Normal(0, 2)  # N(0, 2) distribution

# Integrate x^2 * f(x) dx from -5σ to 5σ with 7 quadrature points
σ = 2
lower, upper = -5σ, 5σ
nodes7, weights7 = lgwt(7, lower, upper)
integral7 = sum(weights7 .* (nodes7.^2) .* pdf.(d, nodes7))
println("a. Integral with 7 points: ", integral7)

# Integrate with 10 quadrature points
nodes10, weights10 = lgwt(10, lower, upper)
integral10 = sum(weights10 .* (nodes10.^2) .* pdf.(d, nodes10))
println("a. Integral with 10 points: ", integral10)

# True variance of N(0, 2) is 4
println("a. True variance: 4")
println("a. Error with 7 points: ", abs(4 - integral7))
println("a. Error with 10 points: ", abs(4 - integral10))

# b. Monte Carlo integration

function monte_carlo_integral(f, a, b, D)
    X = rand(Uniform(a, b), D)
    return (b - a) * mean(f.(X))
end

# Define the integrand
d = Normal(0, 2)
f(x) = x^2 * pdf(d, x)

# Integrate x^2 * f(x) dx from -5σ to 5σ
σ = 2
lower, upper = -5σ, 5σ

# With D = 1,000,000
result_1m = monte_carlo_integral(f, lower, upper, 1_000_000)
println("b. Integral (D = 1,000,000): ", result_1m)

# Integrate x * f(x) dx
g(x) = x * pdf(d, x)
result_mean = monte_carlo_integral(g, lower, upper, 1_000_000)
println("b. Mean (should be close to 0): ", result_mean)

# Integrate f(x) dx
h(x) = pdf(d, x)
result_total_prob = monte_carlo_integral(h, lower, upper, 1_000_000)
println("b. Total probability (should be close to 1): ", result_total_prob)

# With D = 1,000
result_1k = monte_carlo_integral(f, lower, upper, 1_000)
println("b. Integral (D = 1,000): ", result_1k)

println("b. True variance: 4")
println("b. Error with D = 1,000,000: ", abs(4 - result_1m))
println("b. Error with D = 1,000: ", abs(4 - result_1k))

# c. Monte Carlo integration

using Distributions, Random

function monte_carlo_integral(f, a, b, D)
    X = rand(Uniform(a, b), D)
    return (b - a) * mean(f.(X))
end

# Define the distribution
d = Normal(0, 2)

# Set the integration bounds
σ = 2
lower, upper = -5σ, 5σ

# 1. Integrate x^2 * f(x) dx from -5σ to 5σ with D = 1,000,000
f(x) = x^2 * pdf(d, x)
result_variance = monte_carlo_integral(f, lower, upper, 1_000_000)
println("c. Integral of x^2 * f(x) (D = 1,000,000): ", result_variance)
println("   Error: ", abs(4 - result_variance))

# 2. Integrate x * f(x) dx from -5σ to 5σ with D = 1,000,000
g(x) = x * pdf(d, x)
result_mean = monte_carlo_integral(g, lower, upper, 1_000_000)
println("c. Integral of x * f(x) (D = 1,000,000): ", result_mean)
println("   Should be close to 0. Error: ", abs(0 - result_mean))

# 3. Integrate f(x) dx from -5σ to 5σ with D = 1,000,000
h(x) = pdf(d, x)
result_total_prob = monte_carlo_integral(h, lower, upper, 1_000_000)
println("c. Integral of f(x) (D = 1,000,000): ", result_total_prob)
println("   Should be close to 1. Error: ", abs(1 - result_total_prob))

# 4. Repeat the first integral with D = 1,000
result_variance_1k = monte_carlo_integral(f, lower, upper, 1_000)
println("c. Integral of x^2 * f(x) (D = 1,000): ", result_variance_1k)
println("   Error: ", abs(4 - result_variance_1k))

# Comment on the results
println("\nc. Comments on the results:")
println("   - The integral of x^2 * f(x) approximates the variance of the distribution, which should be 4.")
println("   - The integral of x * f(x) approximates the mean of the distribution, which should be 0.")
println("   - The integral of f(x) approximates the total probability, which should be 1.")
println("   - Using D = 1,000,000 gives more accurate results than D = 1,000.")
println("   - The accuracy of the Monte Carlo integration improves as the number of random draws increases.")

# d. Similarity between quadrature and Monte Carlo integration

println("d. Similarity between quadrature and Monte Carlo integration:")

# Quadrature approximation
function quadrature_integral(f, a, b, n)
    nodes, weights = lgwt(n, a, b)
    return sum(weights .* f.(nodes))
end

# Monte Carlo approximation
function monte_carlo_integral(f, a, b, D)
    X = rand(Uniform(a, b), D)
    return (b - a) * mean(f.(X))
end

println("\nQuadrature approximation:")
println("∫ f(x) dx ≈ Σ ωᵢ * f(ξᵢ)")
println("where ωᵢ are quadrature weights and ξᵢ are quadrature nodes")

println("\nMonte Carlo approximation:")
println("∫ f(x) dx ≈ (b - a) * (1/D) * Σ f(Xᵢ)")
println("where D is the number of random draws and Xᵢ ~ U[a, b]")

println("\nSimilarities:")
println("1. Both methods approximate the integral as a weighted sum of function evaluations.")
println("2. In quadrature, the 'weight' is ωᵢ, while in Monte Carlo, it's (b-a)/D for all points.")
println("3. In quadrature, the evaluation points are the predetermined nodes ξᵢ.")
println("   In Monte Carlo, the evaluation points are the random draws Xᵢ.")

println("\nKey differences:")
println("1. Quadrature uses carefully chosen points and weights for optimal accuracy.")
println("2. Monte Carlo uses random sampling with uniform weights.")
println("3. Quadrature often performs better for low-dimensional integrals.")
println("4. Monte Carlo is often preferred for high-dimensional integrals due to its dimension-independent convergence rate.")

# Demonstration
f(x) = x^2 * pdf(Normal(0, 2), x)
a, b = -10, 10

quad_result = quadrature_integral(f, a, b, 10)
mc_result = monte_carlo_integral(f, a, b, 10000)

println("\nDemonstration:")
println("Integral of x^2 * f(x) where f is N(0,2) PDF, from -10 to 10:")
println("Quadrature result (10 points): ", quad_result)
println("Monte Carlo result (10000 points): ", mc_result)
println("True value (variance of N(0,2)): 4")

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
                                         #Question 4#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#                            
using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables, Distributions, ForwardDiff
include("lgwt.jl")

# Load data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS4-mixture/nlsw88t.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)
X = [df.age df.white df.collgrad]
Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
         df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
y = df.occ_code

function mixlogit_quad_with_Z(theta, X, Z, y, R)
    alpha = theta[1:end-2]
    gamma_mean = theta[end-1]
    gamma_sd = exp(theta[end])
    K = size(X, 2)
    J = length(unique(y))
    N = length(y)
    
    bigY = zeros(N, J)
    for j in 1:J
        bigY[:, j] = y .== j
    end
    bigAlpha = [reshape(alpha, K, J-1) zeros(K)]
    
    T = promote_type(eltype(X), eltype(theta))
    nodes, weights = lgwt(R, -4, 4)
    
    ll = 0.0
    for i in 1:N
        prob_i = 0.0
        for r in 1:R
            gamma = gamma_mean + gamma_sd * nodes[r]
            utilities = [X[i,:] ⋅ bigAlpha[:,j] + gamma * (Z[i,j] - Z[i,J]) for j in 1:(J-1)]
            push!(utilities, 0.0)
            probabilities = exp.(utilities) ./ sum(exp.(utilities))
            prob_i += weights[r] * probabilities[y[i]] * pdf(Normal(0, 1), nodes[r])
        end
        ll += log(prob_i)
    end
    
    return -ll
end

function estimate_mixed_logit(X, Z, y, R=7)
    N, K = size(X)
    J = size(Z, 2)
    
    # Use the estimates from Question 1 as starting values
    startvals = [0.05570767876416688, 0.08342649976722213, -2.344887681361976, 0.04500076157943125, 0.7365771540890512, -3.153244238810631, 0.09264606406280998, -0.08417701777996893, -4.273280002738097, 0.023903455659102114, 0.7230648923377259, -3.749393470343111, 0.03608733246865346, -0.6437658344513095, -4.2796847340030375, 0.0853109465190059, -1.1714299392376775, -6.678677013966667, 0.086620198654063, -0.7978777029320784, -4.969132023685069, -0.0941942241795243]
    startvals = vcat(startvals[1:end-1], startvals[end], log(0.1))
    
    result = optimize(theta -> mixlogit_quad_with_Z(theta, X, Z, y, R),
                      startvals,
                      BFGS(),
                      Optim.Options(show_trace = true, iterations = 100000);
                      autodiff = :forward)
    
    estimates = Optim.minimizer(result)
    se = sqrt.(diag(inv(ForwardDiff.hessian(theta -> mixlogit_quad_with_Z(theta, X, Z, y, R), estimates))))
    
    return estimates, se
end

# Run estimation
println("Starting mixed logit estimation with quadrature...")
mixed_estimates, mixed_se = estimate_mixed_logit(X, Z, y)

# Print results
println("\nMixed Logit Estimates (Quadrature):")
K = size(X, 2)
J = size(Z, 2)
for j in 1:(J-1)
    for k in 1:K
        idx = (j-1)*K + k
        println("β_$(j)_$(k): ", round(mixed_estimates[idx], digits=6), " (", round(mixed_se[idx], digits=6), ")")
    end
end
println("γ_mean: ", round(mixed_estimates[end-1], digits=6), " (", round(mixed_se[end-1], digits=6), ")")
println("γ_sd: ", round(exp(mixed_estimates[end]), digits=6), " (", round(exp(mixed_se[end]), digits=6), ")")

# Calculate and print t-statistics
t_stats = mixed_estimates ./ mixed_se
println("\nt-statistics:")
for j in 1:(J-1)
    for k in 1:K
        idx = (j-1)*K + k
        println("β_$(j)_$(k): ", round(t_stats[idx], digits=4))
    end
end
println("γ_mean: ", round(t_stats[end-1], digits=4))
println("γ_sd: ", round(t_stats[end], digits=4))

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
                                             #Question 5#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables, Distributions, ForwardDiff

# Load data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS4-mixture/nlsw88t.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)
X = [df.age df.white df.collgrad]
Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
         df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
y = df.occ_code

function mixlogit_mc_with_Z(theta, X, Z, y, R)
    alpha = theta[1:end-2]
    gamma_mean = theta[end-1]
    gamma_sd = exp(theta[end])
    K = size(X, 2)
    J = length(unique(y))
    N = length(y)
    
    bigY = zeros(N, J)
    for j in 1:J
        bigY[:, j] = y .== j
    end
    bigAlpha = [reshape(alpha, K, J-1) zeros(K)]
    
    ll = 0.0
    for i in 1:N
        prob_i = 0.0
        for r in 1:R
            gamma = gamma_mean + gamma_sd * randn()
            utilities = [X[i,:] ⋅ bigAlpha[:,j] + gamma * (Z[i,j] - Z[i,J]) for j in 1:(J-1)]
            push!(utilities, 0.0)
            probabilities = exp.(utilities) ./ sum(exp.(utilities))
            prob_i += probabilities[y[i]]
        end
        ll += log(prob_i / R)
    end
    
    return -ll
end

function estimate_mixed_logit_mc(X, Z, y, R=1000)
    N, K = size(X)
    J = size(Z, 2)
    
    # Use the estimates from Question 1 as starting values
    startvals = [0.05570767876416688, 0.08342649976722213, -2.344887681361976, 0.04500076157943125, 0.7365771540890512, -3.153244238810631, 0.09264606406280998, -0.08417701777996893, -4.273280002738097, 0.023903455659102114, 0.7230648923377259, -3.749393470343111, 0.03608733246865346, -0.6437658344513095, -4.2796847340030375, 0.0853109465190059, -1.1714299392376775, -6.678677013966667, 0.086620198654063, -0.7978777029320784, -4.969132023685069, -0.0941942241795243]
    startvals = vcat(startvals[1:end-1], startvals[end], log(0.1))
    
    result = optimize(theta -> mixlogit_mc_with_Z(theta, X, Z, y, R),
                      startvals,
                      BFGS(),
                      Optim.Options(show_trace = true, iterations = 100000);
                      autodiff = :forward)
    
    estimates = Optim.minimizer(result)
    se = sqrt.(diag(inv(ForwardDiff.hessian(theta -> mixlogit_mc_with_Z(theta, X, Z, y, R), estimates))))
    
    return estimates, se
end

# Run estimation
println("Starting mixed logit estimation with Monte Carlo...")
mixed_estimates_mc, mixed_se_mc = estimate_mixed_logit_mc(X, Z, y)

# Print results
println("\nMixed Logit Estimates (Monte Carlo):")
K = size(X, 2)
J = size(Z, 2)
for j in 1:(J-1)
    for k in 1:K
        idx = (j-1)*K + k
        println("β_$(j)_$(k): ", round(mixed_estimates_mc[idx], digits=6), " (", round(mixed_se_mc[idx], digits=6), ")")
    end
end
println("γ_mean: ", round(mixed_estimates_mc[end-1], digits=6), " (", round(mixed_se_mc[end-1], digits=6), ")")
println("γ_sd: ", round(exp(mixed_estimates_mc[end]), digits=6), " (", round(exp(mixed_se_mc[end]), digits=6), ")")

# Calculate and print t-statistics
t_stats_mc = mixed_estimates_mc ./ mixed_se_mc
println("\nt-statistics:")
for j in 1:(J-1)
    for k in 1:K
        idx = (j-1)*K + k
        println("β_$(j)_$(k): ", round(t_stats_mc[idx], digits=4))
    end
end
println("γ_mean: ", round(t_stats_mc[end-1], digits=4))
println("γ_sd: ", round(t_stats_mc[end], digits=4))

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
                                               #question 6#
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables, Distributions, ForwardDiff
include("lgwt.jl")

function run_all_estimations()
    # Load data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS4-mixture/nlsw88t.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [df.age df.white df.collgrad]
    Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
             df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
    y = df.occ_code

    # Question 1: Multinomial Logit
    function mlogit_with_Z(theta, X, Z, y)
        alpha = theta[1:end-1]
        gamma = theta[end]
        K = size(X,2)
        J = length(unique(y))
        N = length(y)
        bigY = zeros(N,J)
        for j=1:J
            bigY[:,j] = y.==j
        end
        bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
        
        T = promote_type(eltype(X),eltype(theta))
        num   = zeros(T,N,J)
        dem   = zeros(T,N)
        for j=1:J
            num[:,j] = exp.(X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J])*gamma)
            dem .+= num[:,j]
        end
        
        P = num./repeat(dem,1,J)
        
        loglike = -sum( bigY.*log.(P) )
        
        return loglike
    end

    function estimate_mlogit(X, Z, y)
        K = size(X,2)
        J = length(unique(y))
        initial_params = vcat(zeros(K*(J-1)), 0.0)
        
        result = optimize(theta -> mlogit_with_Z(theta, X, Z, y),
                          initial_params,
                          BFGS(),
                          Optim.Options(show_trace = true, iterations = 10000);
                          autodiff = :forward)
        
        estimates = Optim.minimizer(result)
        se = sqrt.(diag(inv(ForwardDiff.hessian(theta -> mlogit_with_Z(theta, X, Z, y), estimates))))
        
        return estimates, se
    end

    println("Estimating Multinomial Logit...")
    mlogit_estimates, mlogit_se = estimate_mlogit(X, Z, y)
    println("Multinomial Logit Estimation Complete.")
    println("γ estimate: ", mlogit_estimates[end])
    println("γ standard error: ", mlogit_se[end])

    # Question 3: Mixed Logit with Quadrature
    function mixlogit_quad_with_Z(theta, X, Z, y, R)
        alpha = theta[1:end-2]
        gamma_mean = theta[end-1]
        gamma_sd = exp(theta[end])
        K = size(X, 2)
        J = length(unique(y))
        N = length(y)
        
        bigY = zeros(N, J)
        for j in 1:J
            bigY[:, j] = y .== j
        end
        bigAlpha = [reshape(alpha, K, J-1) zeros(K)]
        
        nodes, weights = lgwt(R, -4, 4)
        
        ll = 0.0
        for i in 1:N
            prob_i = 0.0
            for r in 1:R
                gamma = gamma_mean + gamma_sd * nodes[r]
                utilities = [X[i,:] ⋅ bigAlpha[:,j] + gamma * (Z[i,j] - Z[i,J]) for j in 1:(J-1)]
                push!(utilities, 0.0)
                probabilities = exp.(utilities) ./ sum(exp.(utilities))
                prob_i += weights[r] * probabilities[y[i]] * pdf(Normal(0, 1), nodes[r])
            end
            ll += log(prob_i)
        end
        
        return -ll
    end

    function estimate_mixed_logit_quad(X, Z, y, R=7)
        N, K = size(X)
        J = size(Z, 2)
        
        startvals = vcat(mlogit_estimates[1:end-1], mlogit_estimates[end], log(0.1))
        
        result = optimize(theta -> mixlogit_quad_with_Z(theta, X, Z, y, R),
                          startvals,
                          BFGS(),
                          Optim.Options(show_trace = true, iterations = 100000);
                          autodiff = :forward)
        
        estimates = Optim.minimizer(result)
        se = sqrt.(diag(inv(ForwardDiff.hessian(theta -> mixlogit_quad_with_Z(theta, X, Z, y, R), estimates))))
        
        return estimates, se
    end

    println("Estimating Mixed Logit with Quadrature...")
    mixed_quad_estimates, mixed_quad_se = estimate_mixed_logit_quad(X, Z, y)
    println("Mixed Logit with Quadrature Estimation Complete.")
    println("γ_mean estimate: ", mixed_quad_estimates[end-1])
    println("γ_mean standard error: ", mixed_quad_se[end-1])
    println("γ_sd estimate: ", exp(mixed_quad_estimates[end]))
    println("γ_sd standard error: ", exp(mixed_quad_se[end]))

    # Question 5: Mixed Logit with Monte Carlo
    function mixlogit_mc_with_Z(theta, X, Z, y, R)
        alpha = theta[1:end-2]
        gamma_mean = theta[end-1]
        gamma_sd = exp(theta[end])
        K = size(X, 2)
        J = length(unique(y))
        N = length(y)
        
        bigY = zeros(N, J)
        for j in 1:J
            bigY[:, j] = y .== j
        end
        bigAlpha = [reshape(alpha, K, J-1) zeros(K)]
        
        ll = 0.0
        for i in 1:N
            prob_i = 0.0
            for r in 1:R
                gamma = gamma_mean + gamma_sd * randn()
                utilities = [X[i,:] ⋅ bigAlpha[:,j] + gamma * (Z[i,j] - Z[i,J]) for j in 1:(J-1)]
                push!(utilities, 0.0)
                probabilities = exp.(utilities) ./ sum(exp.(utilities))
                prob_i += probabilities[y[i]]
            end
            ll += log(prob_i / R)
        end
        
        return -ll
    end

    function estimate_mixed_logit_mc(X, Z, y, R=1000)
        N, K = size(X)
        J = size(Z, 2)
        
        startvals = vcat(mlogit_estimates[1:end-1], mlogit_estimates[end], log(0.1))
        
        result = optimize(theta -> mixlogit_mc_with_Z(theta, X, Z, y, R),
                          startvals,
                          BFGS(),
                          Optim.Options(show_trace = true, iterations = 100000);
                          autodiff = :forward)
        
        estimates = Optim.minimizer(result)
        se = sqrt.(diag(inv(ForwardDiff.hessian(theta -> mixlogit_mc_with_Z(theta, X, Z, y, R), estimates))))
        
        return estimates, se
    end

    println("Estimating Mixed Logit with Monte Carlo...")
    mixed_mc_estimates, mixed_mc_se = estimate_mixed_logit_mc(X, Z, y)
    println("Mixed Logit with Monte Carlo Estimation Complete.")
    println("γ_mean estimate: ", mixed_mc_estimates[end-1])
    println("γ_mean standard error: ", mixed_mc_se[end-1])
    println("γ_sd estimate: ", exp(mixed_mc_estimates[end]))
    println("γ_sd standard error: ", exp(mixed_mc_se[end]))

    return mlogit_estimates, mlogit_se, mixed_quad_estimates, mixed_quad_se, mixed_mc_estimates, mixed_mc_se
end

# Call the function to run all estimations
mlogit_estimates, mlogit_se, mixed_quad_estimates, mixed_quad_se, mixed_mc_estimates, mixed_mc_se = run_all_estimations()


#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
                                               # Question 7#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables, Distributions, ForwardDiff
include("lgwt.jl")

# Question 1: Multinomial Logit
function mlogit_with_Z(theta, X, Z, y)
    # ... (implementation from Question 1)
end

function estimate_mlogit(X, Z, y)
    # ... (implementation from Question 1)
end

# Question 2: Interpretation
function interpret_gamma(estimate, se)
    t_stat = estimate / se
    p_value = 2 * (1 - cdf(Normal(), abs(t_stat)))
    return p_value < 0.05
end

# Question 4: Mixed Logit with Quadrature
function mixlogit_quad_with_Z(theta, X, Z, y, R)
    # ... (implementation from Question 4)
end

function estimate_mixed_logit_quad(X, Z, y, R=7)
    # ... (implementation from Question 4)
end

# Question 5: Mixed Logit with Monte Carlo
function mixlogit_mc_with_Z(theta, X, Z, y, R)
    # ... (implementation from Question 5)
end

function estimate_mixed_logit_mc(X, Z, y, R=1000)
    # ... (implementation from Question 5)
end

# Question 6: Wrap All Estimations
function run_all_estimations()
    # ... (implementation from Question 6)
end

# Main execution
mlogit_estimates, mlogit_se, mixed_quad_estimates, mixed_quad_se, mixed_mc_estimates, mixed_mc_se = run_all_estimations()

# Unit Tests
using Test

# Question 1 Tests
@testset "Question 1: Multinomial Logit" begin
    X_test = [1.0 0.0; 1.0 1.0]
    Z_test = [0.5 0.3; 0.2 0.4]
    y_test = [1, 2]
    theta_test = [0.5, -0.3, 0.2]

    @test isapprox(mlogit_with_Z(theta_test, X_test, Z_test, y_test), 1.3862943611198906, atol=1e-6)

    estimates, se = estimate_mlogit(X_test, Z_test, y_test)
    @test length(estimates) == 3
    @test length(se) == 3
    @test all(isfinite.(estimates))
    @test all(isfinite.(se))
end

# Question 2 Tests
@testset "Question 2: Interpretation" begin
    @test interpret_gamma(0.5, 0.1)  # Should be significant
    @test !interpret_gamma(0.1, 0.5)  # Should not be significant
end

# Question 3 Tests
@testset "Question 3: Quadrature and Monte Carlo" begin
    function test_quadrature()
        d = Normal(0, 2)
        nodes, weights = lgwt(7, -10, 10)
        integral = sum(weights .* (nodes.^2) .* pdf.(d, nodes))
        return isapprox(integral, 4, atol=1e-2)
    end

    function test_monte_carlo()
        d = Normal(0, 2)
        Random.seed!(123)
        samples = rand(d, 1_000_000)
        integral = mean(samples.^2)
        return isapprox(integral, 4, atol=1e-2)
    end

    @test test_quadrature()
    @test test_monte_carlo()
end

# Question 4 Tests
@testset "Question 4: Mixed Logit with Quadrature" begin
    X_test = [1.0 0.0; 1.0 1.0]
    Z_test = [0.5 0.3; 0.2 0.4]
    y_test = [1, 2]
    theta_test = [0.5, -0.3, 0.2, 0.1]

    @test isapprox(mixlogit_quad_with_Z(theta_test, X_test, Z_test, y_test, 5), 1.3862943611198906, atol=1e-3)

    estimates, se = estimate_mixed_logit_quad(X_test, Z_test, y_test, 5)
    @test length(estimates) == 4
    @test length(se) == 4
    @test all(isfinite.(estimates))
    @test all(isfinite.(se))
end

# Question 5 Tests
@testset "Question 5: Mixed Logit with Monte Carlo" begin
    X_test = [1.0 0.0; 1.0 1.0]
    Z_test = [0.5 0.3; 0.2 0.4]
    y_test = [1, 2]
    theta_test = [0.5, -0.3, 0.2, 0.1]

    Random.seed!(123)  # Set seed for reproducibility
    @test isapprox(mixlogit_mc_with_Z(theta_test, X_test, Z_test, y_test, 1000), 1.3862943611198906, atol=1e-2)

    Random.seed!(123)  # Set seed for reproducibility
    estimates, se = estimate_mixed_logit_mc(X_test, Z_test, y_test, 100)
    @test length(estimates) == 4
    @test length(se) == 4
    @test all(isfinite.(estimates))
    @test all(isfinite.(se))
end