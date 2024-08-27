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

