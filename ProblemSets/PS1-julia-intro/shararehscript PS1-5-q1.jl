using Test, Random, LinearAlgebra, JLD, Statistics

# Function q1 (from question 1)
function q1()
    Random.seed!(1234)
    
    A = 15 * rand(10, 7) .- 5  # Corrected U[-5, 10] distribution
    B = 15 * randn(10, 7) .- 2  # N(-2, 15) distribution
    C = [A[1:5, 1:5] B[1:5, 6:7]]
    D = [A[i,j] <= 0 ? A[i,j] : 0 for i in 1:10, j in 1:7]
    
    return A, B, C, D
end

# Unit tests for q1
@testset "q1 Tests" begin
    A, B, C, D = q1()
    
    @test size(A) == (10, 7)
    @test size(B) == (10, 7)
    @test size(C) == (5, 7)
    @test size(D) == (10, 7)
    
    @test all(A .>= -5) && all(A .<= 10)  # Corrected range
    @test all(D .<= 0)
    @test isapprox(mean(B), -2, atol=2.0)  # Given N(-2, 15), corrected with Statistics import
end

