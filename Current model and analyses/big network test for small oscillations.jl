using DifferentialEquations
using LinearAlgebra
using Random
using Statistics
using Plots
using Printf
using Dates
using Base.Threads # Required for multithreading utilities
using Pkg; Pkg.add("PyPlot")

pyplot()

default(
    titlefont = (15, "sans-serif"), 
    legendfontsize = 10, 
    guidefont = font(6), 
    xtickfont = font(pointsize = 6, family = "sans-serif"), 
    ytickfont = font(pointsize = 6, family = "sans-serif"), 
    guide = "x", 
    grid = false, 
    fontfamily = "sans-serif"
)

# -------------------------------------------------------------------------
# OPTIMIZED ODE SYSTEM
# -------------------------------------------------------------------------
function system_ode!(dx, x, W, t)
    n = length(x)
    @inbounds for i in 1:n
        dot_val = zero(eltype(x)) # Dynamically adapts to Float64 or Dual numbers
        for j in 1:n
            dot_val += W[i,j] * x[j]
        end
        # zero(eltype(x)) safely handles the type inference for the stiff solver
        dx[i] = -x[i] + max(zero(eltype(x)), dot_val + 1.0)
    end
end

# -------------------------------------------------------------------------
# OPTIMIZED WEIGHT MATRIX GENERATOR
# -------------------------------------------------------------------------
function generate_weight_matrix!(W::Matrix{Float64}, p::Float64, q::Float64, E_PARAM::Float64, D_PARAM::Float64)
    w_excited = -1.0 + E_PARAM
    w_inhibited = -1.0 - D_PARAM
    
    p1 = 1.0 - p
    p2 = p1 + p * (1.0 - q) / 2.0
    p3 = p2 + p * (1.0 - q) / 2.0
    
    n = size(W, 1)
    fill!(W, 0.0)
    
    @inbounds for j in 1:n
        for i in 1:(j-1)
            r = rand()
            if r <= p1
                W[i, j] = w_inhibited
                W[j, i] = w_inhibited
            elseif r <= p2
                W[i, j] = w_inhibited
                W[j, i] = w_excited
            elseif r <= p3
                W[i, j] = w_excited
                W[j, i] = w_inhibited
            else
                W[i, j] = w_excited
                W[j, i] = w_excited
            end
        end
    end
    return W
end

E_PARAM = 0.25
D_PARAM = 0.5
MATRIX_SIZE = 500
P_VALS = collect(range(0.0, 1.0, length=11))
Q_VALS = collect(range(0.0, 1.0, length=11))
NUM_TRIALS = 15

TIME_SPAN = (0.0, 600.0)
CONVERGENCE_TOL = 0.001/MATRIX_SIZE
ACTIVE_THRESHOLD = 1e-7 # Lowered to noise floor to handle O(1/N) steady states
DERIV_ERROR_THRESHOLD = 1e-3

W = zeros(Float64, MATRIX_SIZE, MATRIX_SIZE)
x0 = zeros(Float64, MATRIX_SIZE)
prob_base = ODEProblem(system_ode!, x0, TIME_SPAN, W)

generate_weight_matrix!(W, 1, 0, E_PARAM, D_PARAM)
rand!(x0) 

W_copy = copy(W)
prob = remake(prob_base, u0=copy(x0), p=W_copy)

# Auto-switching stiff solver with tightened tolerances
sol = solve(prob, FBDF(), saveat=save_times, reltol=1e-5, abstol=1e-8)

final_state = sol.u[end]

if length(sol.u) > 1
    recent_traj = reduce(hcat, sol.u[2:end]) 
    
    # 1. CONVERGENCE CHECK
    # max_diff = 0.0
    # for step in 2:length(sol.u)
    #     diff = maximum(abs.(final_state .- sol.u[step]))
    #     max_diff = max(max_diff, diff)
    # end
    # trial_convergence[trial] = max_diff <= CONVERGENCE_TOL ? 1 : 0

    # 1. CONVERGENCE CHECK (Standard Deviation based)
    # A fixed point should have zero variance over the final window.
    # We check if the maximum standard deviation of any neuron is below tolerance.
    traj_std = std(recent_traj, dims=2) |> vec
    max_std = maximum(traj_std)
    trial_convergence[trial] = max_std <= CONVERGENCE_TOL ? 1 : 0
    
    # 2. ACTIVE NEURON CHECK
    recent_max_state = maximum(recent_traj, dims=2) |> vec
    active_mask = recent_max_state .> ACTIVE_THRESHOLD
    trial_active_counts[trial] = sum(active_mask)
    
    # 3. REGIME CLASSIFICATION
    dx = -recent_traj .+ max.(0.0, W_copy * recent_traj .+ 1.0)
    
    max_dx = maximum(dx, dims=2) |> vec
    min_dx = minimum(dx, dims=2) |> vec
    diff_dx = abs.(max_dx .- min_dx)
    
    fp_mask = diff_dx .< DERIV_ERROR_THRESHOLD
    nl_mask = .~fp_mask
    
    active_fp_count = sum(active_mask .& fp_mask)
    active_nl_count = sum(active_mask .& nl_mask)
    
    if active_nl_count > 0
        push!(trial_nl_counts, active_nl_count)
    elseif active_fp_count > 0
        push!(trial_fp_counts, active_fp_count)
    end
end
