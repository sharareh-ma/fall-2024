using Test
using Random

# Define q1 function
function q1()
    Random.seed!(1234)  # Ensures reproducibility
    
    # A: A 10x7 matrix with random numbers between -5 and 10
    A = 15 * rand(10, 7) .- 5
    
    # B: A 10x7 matrix with random numbers from a normal distribution with mean -2 and std 15
    B = 15 * randn(10, 7) .- 2
    
    # C: A 5x7 matrix made from parts of A and B
    C = [A[1:5, 1:5] B[1:5, 6:7]]
    
    # D: A 10x7 matrix where each element is A[i,j] if A[i,j] <= 0, otherwise 0
    D = [A[i,j] <= 0 ? A[i,j] : 0 for i in 1:10, j in 1:7]
    
    return A, B, C, D
end

# Define q2 function
function q2(A::Matrix, B::Matrix, C::Matrix)
    # Element-by-element product using a comprehension
    AB = [A[i, j] * B[i, j] for i in 1:size(A, 1), j in 1:size(A, 2)]
    
    # Element-by-element product without comprehension
    AB2 = A .* B
    
    # Filter elements of C between -5 and 5
    Cprime = [x for x in C if -5 <= x <= 5]
    
    # Filter elements without a loop or comprehension
    Cprime2 = filter(x -> -5 <= x <= 5, C)
    
    return AB, AB2, Cprime, Cprime2
end

# Run q1 to generate A, B, C, D
A, B, C, D = q1()

# Unit tests for q2
@testset "q2 Tests" begin
    AB, AB2, Cprime, Cprime2 = q2(A, B, C)
    
    # Test 1: Ensure AB and AB2 are identical
    @test AB == AB2
    
    # Test 2: Ensure AB and AB2 have the correct size
    @test size(AB) == size(A)
    @test size(AB2) == size(A)
    
    # Test 3: Ensure Cprime and Cprime2 are identical
    @test Cprime == Cprime2
    
    # Test 4: Ensure all elements in Cprime fall within the range [-5, 5]
    @test all(-5 .<= Cprime .<= 5)
end


