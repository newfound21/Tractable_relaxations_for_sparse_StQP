##############################################################
# Authors: Yuzhou Qiu and E. Alper Yildirim
#
# This code can be used to reproduce the computational experiments 
# on COP instances. Seed is initialised from 0 but only results from 
# seed 1 onward are reported in the manuscript.
#
################################################################

using JuMP
using LinearAlgebra
using Distributions
using Random
using DataFrames
using CSV
using DelimitedFiles

include("GenerateOptSol.jl")
include("Generate_Q_COP.jl")
include("Gurobi_P1rho.jl")
include("Gurobi_P2rho.jl")
include("D1A-FirstDNN-Relaxation.jl")
include("D1B-FirstDNNRelaxation.jl")
include("D2A-SecondDNN-Relaxation.jl")
include("D2B-SecondDNN-Relaxation.jl")

nset = [25, 50]

rho0ratioset = [1 / 4, 1 / 2, 3 / 4]

rhoratioset = [1 / 4, 1 / 2, 3 / 4]

seedset = 0:25

timelimit = 600

for n in nset

    df_for_each_rho = DataFrame(n=[],
        rho0=[],
        seed=[],
        rho=[],
        solution_time_P1=[],
        solution_time_P2=[],
        solution_time_D1A=[],
        solution_time_D1B=[],
        solution_time_D2A=[],
        solution_time_D2B=[],
        optimal_val_P1=[],
        optimal_val_P2=[],
        optimal_val_D1A=[],
        optimal_val_D1B=[],
        optimal_val_D2A=[],
        optimal_val_D2B=[],
        objective_bound_P1=[],
        objective_bound_P2=[],
        objective_bound_D1A=[],
        objective_bound_D1B=[],
        objective_bound_D2A=[],
        objective_bound_D2B=[],
        relative_gap_P1=[],
        relative_gap_P2=[],
        relative_gap_D1A=[],
        relative_gap_D1B=[],
        relative_gap_D2A=[],
        relative_gap_D2B=[],
        termination_status_P1=[],
        termination_status_P2=[],
        termination_status_D1A=[],
        termination_status_D1B=[],
        termination_status_D2A=[],
        termination_status_D2B=[],
    )

    println("n = ", n)

    for rho0ratio in rho0ratioset

        println("rho0ratio = ", rho0ratio)

        for seed in seedset

            println("seed = ", seed)

            Random.seed!(seed)

            rho_0 = round(Int, rho0ratio * n)

            x_hat = Generate(n, rho_0, seed)

            Q_1 = Generate_Q_COP(seed, x_hat)

            for rhoratio in rhoratioset

                println("rhoratio = ", rhoratio)

                rho = round(Int, rho_0 * rhoratio)

                res_P1 = Gurobi_P1rho(Q_1, rho, timelimit)

                res_P2 = Gurobi_P2rho(Q_1, rho, timelimit)

                res_D1A = D1ADNN(Q_1, rho, timelimit)

                res_D1B = D1BDNN(Q_1, rho, timelimit)

                res_D2A = D2ADNN(Q_1, rho, timelimit)

                res_D2B = D2BDNN(Q_1, rho, timelimit)

                current_outputs = [res_P1[1], res_P2[1], res_D1A[1], res_D1B[1], res_D2A[1], res_D2B[1], res_P1[2], res_P2[2], res_D1A[2], res_D1B[2], res_D2A[2], res_D2B[2], res_P1[3], res_P2[3], res_D1A[3], res_D1B[3], res_D2A[3], res_D2B[3], 100 * res_P1[4], 100 * res_P2[4], 100 * res_D1A[4], 100 * res_D1B[4], 100 * res_D2A[4], 100 * res_D2B[4], res_P1[5], res_P2[5], res_D1A[5], res_D1B[5], res_D2A[5], res_D2B[5]]

                current_vec = [n; rho_0; seed; rho; current_outputs]

                push!(df_for_each_rho, current_vec)

            end

        end

    end

    println("n = ", n)

    CSV.write("./n" * string(n) * "-COP.csv", df_for_each_rho)

end


