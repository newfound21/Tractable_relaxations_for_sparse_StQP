##############################################################
# Authors: Yuzhou Qiu and E. Alper Yildirim
#
# This code solves the DNN relaxation (D2B) in the manuscript.
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

function D2BDNN(Q0, k, timelimit)

    n = size(Q0,1)

    #avoid non-symmetric input
    Q0 = (Q0'+Q0) / 2

    model_QCQP_SDP = Model(Mosek.Optimizer)

    set_time_limit_sec(model_QCQP_SDP, timelimit)

    # variable
    # [1 x^T u^T
    #  x S^xx ...
    #  u  ... S^uu]


    @variable(model_QCQP_SDP, X[1:(2n+1), 1:(2n+1)], PSD)

    @objective(model_QCQP_SDP, Min, dot(Q0,X[2:(n+1), 2:(n+1)]))

    #nonnegativity
    @constraint(model_QCQP_SDP, X[1:(2n+1), 1:(2n+1)] .>= 0)
    
    # e^T x = 1
    @constraint(model_QCQP_SDP, sum(X[2:(n+1), 1]) == 1)

    # e^T u = k
    @constraint(model_QCQP_SDP, sum(X[(n+2):(2n+1), 1]) == k)

    # <E, S^xx> = 1
    @constraint(model_QCQP_SDP, sum(X[2:(n+1), 2:(n+1)]) == 1)

    # <E, S^uu> = k^2
    @constraint(model_QCQP_SDP, sum(X[(n+2):(2n+1), (n+2):(2n+1)]) == k * k)

    # diag(S^uu)  = u
    @constraint(model_QCQP_SDP, diag(X[(n+2):(2n+1), (n+2):(2n+1)]) .== X[(n+2):(2n+1), 1])

    # diag(S^xu) = x
    @constraint(model_QCQP_SDP, diag(X[2:(n+1), (n+2):(2n+1)]) .== X[2:(n+1), 1])

    # x e^T - S^xu >= 0
    @constraint(model_QCQP_SDP, X[2:(n+1), 1] * ones(1,n) - X[2:(n+1), (n+2):(2n+1)] .>= 0)

    # e e^T - e u^T - u e^T + S^uu >= 0    
    @constraint(model_QCQP_SDP, ones(n,n) .- ones(n,1) * X[(n+2):(2n+1),1]' .- X[(n+2):(2n+1), 1] * ones(1,n) .+ X[(n+2):(2n+1), (n+2):(2n+1)] .>= 0)

    # u e^T - S^uu >= 0
    @constraint(model_QCQP_SDP, X[(n+2):(2n+1), 1] * ones(1,n) - X[(n+2):(2n+1), (n+2):(2n+1)] .>= 0)

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
