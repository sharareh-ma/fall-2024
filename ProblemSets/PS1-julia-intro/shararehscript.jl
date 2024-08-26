# Import necessary packages
using Random, Distributions, LinearAlgebra, JLD2, DataFrames, CSV

# Define the function q1()
function q1()

    # (a) Create the matrices with specified distributions
    Random.seed!(1234)
    
    # i. A10×7 - random numbers distributed U [−5, 10]
    A = -5 .+ 15 .* rand(10, 7)
    
    # ii. B10×7 - random numbers distributed N (−2, 15)
    B = rand(Normal(-2, 15), 10, 7)
    
    # iii. C5×7 - first 5 rows and first 5 columns of A and the last two columns and first 5 rows of B
    C = hcat(A[1:5, 1:5], B[1:5, 6:7])
    
    # iv. D10×7 - where Di,j = Ai,j if Ai,j ≤ 0, or 0 otherwise
    D = [A[i,j] <= 0 ? A[i,j] : 0 for i in 1:10, j in 1:7]
    
    # (b) Number of elements in A
    num_elements_A = length(A)
    println("Number of elements in A: ", num_elements_A)
    
    # (c) Number of unique elements in D
    num_unique_elements_D = length(unique(D))
    println("Number of unique elements in D: ", num_unique_elements_D)
    
    # (d) Create E as the ‘vec’ operator applied to B using reshape()
    E = reshape(B, :, 1)
    # An easier way would be to use the `vec()` function directly
    E_easy = vec(B)
    
    # (e) Create F which is 3D and contains A in the first column and B in the second column
    F = cat(A, B, dims=3)
    
    # (f) Use permutedims() to twist F to F2×10×7 and save as F
    F = permutedims(F, (3,1,2))
    
    # (g) Create G by applying the Kronecker product to each slice of F
    G_slices = []
    for i in 1:size(F, 3)
        G_slice = kron(C, F[:, :, i])
        push!(G_slices, G_slice)
    end
    G = cat(G_slices..., dims=3)

    # (h) Save A, B, C, D, E, F, G to a .jld file
    @save "matrixpractice.jld2" A B C D E F G
    
    # (i) Save only A, B, C, and D to a .jld file
    @save "firstmatrix.jld2" A B C D
    
    # (j) Export C as a .csv file called Cmatrix
    C_df = DataFrame(C, :auto)
    CSV.write("Cmatrix.csv", C_df)
    
    # (k) Export D as a tab-delimited .dat file called Dmatrix
    D_df = DataFrame(D, :auto)
    open("Dmatrix.dat", "w") do file
        write(file, string(D_df, "\t"))
    end
    
    return A, B, C, D
end

# Call the function and assign the returned values
A, B, C, D = q1()
