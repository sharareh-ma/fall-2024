# Import necessary packages
import Pkg

# Ensure required packages are installed
Pkg.add("Distributions")
Pkg.add("JLD2")

using Random, Distributions, JLD2

# Set the random seed for reproducibility
Random.seed!(1234)

# Create Matrix A: 10×7 with random numbers distributed U[-5, 10]
A = rand(Uniform(-5, 10), 10, 7)

# Create Matrix B: 10×7 with random numbers distributed N(-2, 15)
B = rand(Normal(-2, 15), 10, 7)

# Create Matrix C: 5×7 from parts of A and B
C = hcat(A[1:5, 1:5], B[1:5, 6:7])

# Create Matrix D: 10×7 where Di,j = Ai,j if Ai,j ≤ 0, or 0 otherwise
D = A .* (A .<= 0)

# Save the matrices to a .jld2 file with explicit names using JLD2
@JLD2.save "/Users/sanazma/fall-2024-5/ProblemSets/PS1-julia-intro/firstmatrix.jld2" A B C D

println("firstmatrix.jld2 has been successfully created.")




