##############################################################
# Authors: Yuzhou Qiu and E. Alper Yildirim
#
# This code solves the DNN relaxation (D2A) in the manuscript.
#
# Inputs:
# 
# Q0: an n x n symmetric matrix
#
# k: sparsity parameter rho
#
# timelimit: time limit in seconds
#
# Outputs:
#
# solution time, best objective function value, best lower bound,
# relative gap,termination status
#
################################################################

using JuMP
using LinearAlgebra
using MosekTools

function D2ADNN(Q0, k, timelimit)

    n = size(Q0,1)

    #avoid non-symmetric input
    Q0 = (Q0'+Q0) / 2

    model_QCQP_SDP = Model(Mosek.Optimizer)

    # variable
    # [1 x^T u^T v^T
    #  x Y^xx ...
    #  u      ...
    #  v      ... Y^vv]

    set_time_limit_sec(model_QCQP_SDP, timelimit)

    @variable(model_QCQP_SDP, X[1:(3n+1), 1:(3n+1)], PSD)

    @objective(model_QCQP_SDP, Min, dot(Q0,X[2:(n+1), 2:(n+1)]))

    #nonnegativity
    @constraint(model_QCQP_SDP, X[1:(3n+1), 1:(3n+1)] .>= 0)
    
    # e^T x = 1
    @constraint(model_QCQP_SDP, sum(X[2:(n+1), 1]) == 1)

    # e^T u = k
    @constraint(model_QCQP_SDP, sum(X[(n+2):(2n+1), 1]) == k)

    # u + v = e
    @constraint(model_QCQP_SDP,  X[(n+2):(2n+1), 1] + X[(2n+2):(3n+1), 1] .== ones(n))

    # <E, Y^xx> = 1
    @constraint(model_QCQP_SDP, sum(X[2:(n+1), 2:(n+1)]) == 1)

    # <E, Y^uu> = k^2
    @constraint(model_QCQP_SDP, sum(X[(n+2):(2n+1), (n+2):(2n+1)]) == k * k)

    # diag(Y^uu)  = u
    @constraint(model_QCQP_SDP, diag(X[(n+2):(2n+1), (n+2):(2n+1)]) .== X[(n+2):(2n+1), 1])

    # e^T diag(Y^xv) = 0
    @constraint(model_QCQP_SDP, sum(diag(X[2:(n+1), (2n+2):(3n+1)])) == 0)

    # diag(Y^uu) + 2 diag(Y^uv) + diag(Y^vv) = e
    @constraint(model_QCQP_SDP, diag(X[(n+2):(2n+1), (n+2):(2n+1)]) + 2 * diag(X[(n+2):(2n+1), (2n+2):(3n+1)]) + diag(X[(2n+2):(3n+1), (2n+2):(3n+1)]) .== ones(n))

    @constraint(model_QCQP_SDP, X[1,1] == 1)

    JuMP.optimize!(model_QCQP_SDP)

    println()

    println("Solution status: ",termination_status(model_QCQP_SDP))

    println()

    println("Best objective function value : ",objective_value(model_QCQP_SDP))

    println()

    println("Best feasible solution: ",value.(X[1, 2:(n + 1)]))

    println()

    println("Best lower bound: ",objective_bound(model_QCQP_SDP))

    println()

    println("Relative gap: ",relative_gap(model_QCQP_SDP))

    println()

    println("Solution time: ",solve_time(model_QCQP_SDP))

    println()

    return [solve_time(model_QCQP_SDP),objective_value(model_QCQP_SDP),objective_bound(model_QCQP_SDP),relative_gap(model_QCQP_SDP),termination_status(model_QCQP_SDP)] 

end
