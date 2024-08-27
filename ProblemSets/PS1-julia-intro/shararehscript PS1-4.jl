# Import necessary packages
using JLD2, LinearAlgebra, Statistics, CSV, DataFrames

# Define the function q4
function q4()
    # (a) Load firstmatrix.jld2 using JLD2
    JLD2.@load "/Users/sanazma/fall-2024-5/ProblemSets/PS1-julia-intro/firstmatrix.jld2" A B C D
    
    # (b) Define matrixops function
    """
    matrixops performs three operations:
    1. Element-by-element product of matrices A and B.
    2. Matrix product of the transpose of A and B (A'B).
    3. Sum of all elements in A + B.
    """
    function matrixops(A::AbstractMatrix, B::AbstractMatrix)
        # (e) Error if matrices A and B are not the same size
        if size(A) != size(B)
            error("inputs must have the same size.")
        end

        # (b.i) Element-by-element product
        elem_prod = A .* B

        # (b.ii) Matrix product of A'B
        mat_prod = transpose(A) * B

        # (b.iii) Sum of all elements in A + B
        sum_all = sum(A + B)

        return elem_prod, mat_prod, sum_all
    end
    
    # (d) Evaluate matrixops using A and B
    println("Evaluating matrixops with A and B:")
    println(matrixops(A, B))
    
    # (f) Evaluate matrixops using C and D
    if size(C) == size(D)
        println("Evaluating matrixops with C and D:")
        println(matrixops(C, D))
    else
        println("Skipping matrixops with C and D due to size mismatch.")
    end
    
    # (g) Evaluate matrixops using ttl_exp and wage from nlsw88_processed.csv
    println("Evaluating matrixops with ttl_exp and wage from nlsw88_processed.csv:")

    # Correctly define the file_path variable
    file_path = "/Users/sanazma/fall-2024-5/ProblemSets/PS1-julia-intro/nlsw88_processed.csv"

    if !isfile(file_path)
        println("Error: The file $file_path does not exist.")
        return
    end
    
    nlsw88 = CSV.File(file_path) |> DataFrame
    ttl_exp = convert(Array, nlsw88.ttl_exp)
    wage = convert(Array, nlsw88.wage)

    try
        println(matrixops(ttl_exp, wage))
    catch e
        println("Error: ", e)
    end
end

# Call the function q4 at the end of the script
q4()


    
  




