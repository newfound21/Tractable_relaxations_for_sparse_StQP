##############################################################
# Authors: Yuzhou Qiu and E. Alper Yildirim
#
# This code generates a PSD instance such that x_hat is the unique optimal 
# solution of the StQP without the sparsity constraint.
#
# Inputs:
# 
# seed_n: seed of the random number generator
#
# x_hat: designated unique optimal solution of the StQP without the sparsity constraint
#
# Outputs:
#
# Q_0: an n x n symmetric matrix
#
################################################################

using LinearAlgebra
using Distributions
using Random
using JuMP
using LinearAlgebra
using MosekTools

function Generate_Q_PSD(seed_n, x_hat)

    # initialise the seed of the random number generator
    Random.seed!(seed_n)

    n = size(x_hat,1)

    # find position of zeros and nonzeros of x_hat

    indexA = []

    indexB = []

    for i = 1:n

        if x_hat[i] > 10^(-8)

            push!(indexA, i)

        else

            push!(indexB, i)

        end

    end

    nonzero_num = size(indexA, 1)

    zero_num = size(indexB, 1)

    perm = [indexA; indexB]

    # x_hat[perm] = [x_P; 0]

    # Generate R with uniformly distributed eigenvalues in [0,3]

    eig_matrix = diagm(rand(Uniform(0,3), n))

    A = rand(-5:5,n,n)
    
    M,R0 = qr(A)
    
    R = M * eig_matrix * M'

    # Q_1 before reordering

    Q_1 = (diagm(ones(n)) - ones(n) * x_hat[perm]') * R * (diagm(ones(n)) - ones(n) * x_hat[perm]')'

    invperm = sortperm(perm)

    # Q_1 reordered

    Q_1 = Q_1[invperm, invperm]
    
    return Q_1
end
