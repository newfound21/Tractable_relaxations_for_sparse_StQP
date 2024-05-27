##############################################################
# Authors: Yuzhou Qiu and E. Alper Yildirim
#
# This code solves the MIQP formulation (P2) in the manuscript.
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
using Gurobi

function Gurobi_P2rho(Q0, k, timelimit)

    n = size(Q0,1)

    Q0 = (Q0'+Q0) / 2

    model_sparse_QP = Model(Gurobi.Optimizer)

    set_time_limit_sec(model_sparse_QP, timelimit)

    set_optimizer_attribute(model_sparse_QP, "NonConvex", 2)

    @variable(model_sparse_QP, x[1:n])

    @variable(model_sparse_QP, v[1:n], Bin)

    @objective(model_sparse_QP, Min,  dot(Q0*x, x))

    @constraint(model_sparse_QP, sum(x) == 1)

    @constraint(model_sparse_QP, sum(v) == n - k)

    @constraint(model_sparse_QP, dot(x,v) == 0)

    @constraint(model_sparse_QP, x .>= 0)

    JuMP.optimize!(model_sparse_QP)

    println()

    println("Solution status: ",termination_status(model_sparse_QP))

    println()

    println("Best objective function value : ",objective_value(model_sparse_QP))

    println()

    println("Best feasible solution: ",value.(x))

    println()

    println("Best lower bound: ",objective_bound(model_sparse_QP))

    println()

    println("Relative gap: ",relative_gap(model_sparse_QP))

    println()

    println("Solution time: ",solve_time(model_sparse_QP))

    println()

    return [solve_time(model_sparse_QP),objective_value(model_sparse_QP),objective_bound(model_sparse_QP),relative_gap(model_sparse_QP),termination_status(model_sparse_QP)] 

end
