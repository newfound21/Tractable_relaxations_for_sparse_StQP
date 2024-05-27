##############################################################
# Authors: Yuzhou Qiu and E. Alper Yildirim
#
# This code randomly generates a feasible solution x_hat of StQP without the sparsity constraint
# with specified cardinality, while ensuring that positive components are
# not too small.
#
# Inputs:
# 
# n: dimension
#
# rho_0: sparsity of the designated optimal solution
#
# seed: seed of the random number generator
#
# Outputs:
#
# feasible solution x_hat
#
################################################################

using LinearAlgebra
using Distributions
using Random

function Generate(n, rho_0, seed)

    Random.seed!(seed)

    #

    # ensure that positive components are not too small
    
    flag = true

    while flag

        x_hat = rand(Uniform(0,1), rho_0)

        x_hat = x_hat/sum(x_hat)

        if minimum(x_hat) >= 10^(-8)

            flag = false

        end

    end

    #add zero components
            
    x_hat = [x_hat;zeros(n - rho_0)]

    # shuffle x_hat
            
    x_hat = shuffle(x_hat)

    return x_hat

end