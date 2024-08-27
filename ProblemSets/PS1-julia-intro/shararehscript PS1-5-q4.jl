using Test, LinearAlgebra

# Define matrixops function to handle matrices and vectors
function matrixops(A::AbstractArray, B::AbstractArray)
    if size(A) != size(B)
        error("inputs must have the same size.")
    end

    if ndims(A) == 1
        elem_prod = A .* B
        mat_prod = dot(A, B)  # Scalar result for vectors
        sum_all = sum(A + B)
    else
        elem_prod = A .* B
        mat_prod = transpose(A) * B
        sum_all = sum(A + B)
    end

    return elem_prod, mat_prod, sum_all
end

# Unit tests for matrixops
@testset "matrixops Tests" begin
    # Test 1: Regular matrices
    A = [1 2 3; 4 5 6; 7 8 9]
    B = [9 8 7; 6 5 4; 3 2 1]

    elem_prod, mat_prod, sum_all = matrixops(A, B)

    @test elem_prod == [9 16 21; 24 25 24; 21 16 9]
    @test mat_prod == [54 42 30; 72 57 42; 90 72 54]
    @test sum_all == 90

    # Test 2: Mismatched matrix sizes
    A = [1 2 3; 4 5 6]
    B = [7 8 9]

    @test_throws ErrorException matrixops(A, B)

    # Test 3: Empty matrices
    A = reshape(Float64[], 1, 0)
    B = reshape(Float64[], 1, 0)

    elem_prod, mat_prod, sum_all = matrixops(A, B)

    @test elem_prod == reshape(Float64[], 1, 0)
    @test mat_prod == reshape(Float64[], 0, 0)
    @test sum_all == 0.0

    # Test 4: Vectors instead of matrices
    A_vec = [1.0, 2.0, 3.0]
    B_vec = [4.0, 5.0, 6.0]

    elem_prod, mat_prod, sum_all = matrixops(A_vec, B_vec)

    @test vec(elem_prod) == [4.0, 10.0, 18.0]  # Convert result to vector for comparison
    @test mat_prod == 32.0  # Scalar dot product result
    @test sum_all == 21.0
end
