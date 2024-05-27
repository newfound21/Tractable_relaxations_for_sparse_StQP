##############################################################
# Authors: Yuzhou Qiu and E. Alper Yildirim
#
# This code generates a COP instance such that x_hat is the unique optimal 
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

function Generate_Q_COP(seed_n, x_hat)

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

    if (zero_num < 5)

        error("x_hat should have at least 5 zero components!")

    end

    perm = [indexA; indexB]

    # x_hat[perm] = [x_P; 0]


    # Generate R_BB

    # 1. H is the Horn matrix
    H = [1 -1 1 1 -1;
        -1 1 -1 1 1;
        1 -1 1 -1 1;
        1 1 -1 1 -1;
        -1 1 1 -1 1]

    # nonnegative C with uniformly distributed entries in [0,1]
    C = rand(Uniform(0,1),zero_num - 5,5)

    # generate another positive semidefinite B with uniformly distributed eigenvalues in [0,3]
    eig_matrix = diagm(rand(Uniform(0,3), zero_num - 5))

    A = rand(-5:5,zero_num - 5,zero_num - 5)
    
    M,U = qr(A)
    
    B = M * eig_matrix * M'

    R_BB = [B C;
            C' H]





    # generate a positive definite R_AA with uniformly distributed eigenvalues in [0,3]

    eig_matrix = diagm(rand(Uniform(0,3), nonzero_num))
    A = rand(-5:5,nonzero_num,nonzero_num)
    M,R = qr(A)
    R_AA = M * eig_matrix * M'


    # compute epsilon (err)
    F = [7 4.32 0 0 4.32;
         4.32 7 4.32 0 0;
         0 4.32 7 4.32 0;
         0 0 4.32 7 4.32;
         4.32 0 0 4.32 7]     

    mu = sum(F)

    delta = -dot(H,F)

    err = delta/mu

    
    # scale to R_AA ensure that ||R_AA|| < err

    err = (0.99) * err

    R_AA = R_AA/opnorm(R_AA, 2) * err


    # Q_1 before reordering

    P = [R_AA zeros(nonzero_num, zero_num);
        zeros(zero_num, nonzero_num) R_BB]

    Q_1 = (diagm(ones(n)) - ones(n) * x_hat[perm]') * P * (diagm(ones(n)) - ones(n) * x_hat[perm]')'

    invperm = sortperm(perm)

    # Q_1 reordered

    Q_1 = Q_1[invperm, invperm]
    
    return Q_1

end
