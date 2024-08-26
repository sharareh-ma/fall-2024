# Import necessary packages
using Random, Distributions, LinearAlgebra

# Define the function q2() that will perform all the tasks
function q2(A, B, C)
    # (a) Element-by-element product of A and B
    AB = [A[i, j] * B[i, j] for i in 1:size(A, 1), j in 1:size(A, 2)]
    AB2 = A .* B  # This performs element-wise multiplication without a loop
    
    println("AB matrix calculated using comprehension:")
    println(AB)
    
    println("AB2 matrix calculated without loop/comprehension:")
    println(AB2)
    
    # (b) Create Cprime vector that contains only elements of C between -5 and 5
    Cprime = [C[i, j] for i in 1:size(C, 1), j in 1:size(C, 2) if -5 <= C[i, j] <= 5]
    Cprime2 = vec(C[(C .>= -5) .& (C .<= 5)])  # Without loop

    println("Cprime vector calculated using loop:")
    println(Cprime)
    
    println("Cprime2 vector calculated without loop:")
    println(Cprime2)
    
    # (c) Create a 3D array X of dimension N×K×T where N=15,169, K=6, T=5
    N, K, T = 15169, 6, 5
    X = Array{Float64}(undef, N, K, T)

    for t in 1:T
        X[:, 1, t] = ones(N)  # Intercept
        X[:, 2, t] = rand(Bernoulli(0.75 * (6 - t) / 5), N)  # Dummy variable
        X[:, 3, t] = rand(Normal(15 + t - 1, 5 * (t - 1)), N)  # Continuous normal variable 1
        X[:, 4, t] = rand(Normal(π * (6 - t) / 3, 1 / exp(1)), N)  # Continuous normal variable 2
        X[:, 5, t] = rand(Binomial(20, 0.6), N)  # Discrete variable 1
        X[:, 6, t] = rand(Binomial(20, 0.5), N)  # Discrete variable 2
    end

    println("3D array X:")
    println(X)
    
    # (d) Create a matrix β which is K × T
    β = [t == 1 ? 1 + 0.25 * (i - 1) : 
          t == 2 ? log(t) :
          t == 3 ? -sqrt(t) :
          t == 4 ? exp(t) - exp(t + 1) :
          t == 5 ? t :
          t == 6 ? t / 3 : 0 for i in 1:K, t in 1:T]
    
    println("Matrix β:")
    println(β)

    # (e) Create matrix Y which is N × T defined by Yt = Xt * βt + εt, εt ∼ N(0, σ=0.36)
    Y = Array{Float64}(undef, N, T)
    σ = 0.36
    
    for t in 1:T
        εt = rand(Normal(0, σ), N)
        Y[:, t] = X[:, :, t] * β[:, t] + εt
    end

    println("Matrix Y:")
    println(Y)
    
    # Return nothing as specified
    return nothing
end

# Define the function q1() if needed
function q1()
    Random.seed!(1234)
    
    A = -5 .+ 15 .* rand(10, 7)
    B = rand(Normal(-2, 15), 10, 7)
    C = hcat(A[1:5, 1:5], B[1:5, 6:7])
    
    println("Matrix A:")
    println(A)
    
    println("Matrix B:")
    println(B)
    
    println("Matrix C:")
    println(C)
    
    return A, B, C
end

# Call q1() to generate A, B, C
A, B, C = q1()

# Call q2() with A, B, C
q2(A, B, C)
