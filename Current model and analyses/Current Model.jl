using DifferentialEquations
using LinearAlgebra
using Random
using Statistics
using Plots
using Printf
using Dates

# -------------------------------------------------------------------------
# OPTIMIZED ODE SYSTEM
# In-place, non-allocating version of the network dynamics
# -------------------------------------------------------------------------
function system_ode!(dx, x, W, t)
    mul!(dx, W, x) # In-place matrix-vector multiplication (dx = W * x)
    @inbounds @simd for i in eachindex(dx)
        dx[i] = -x[i] + max(0.0, dx[i] + 1.0)
    end
end

# -------------------------------------------------------------------------
# OPTIMIZED WEIGHT MATRIX GENERATOR
# Avoids allocating intermediate arrays or probability weights 
# -------------------------------------------------------------------------
function generate_weight_matrix!(W::Matrix{Float64}, p::Float64, q::Float64, E_PARAM::Float64, D_PARAM::Float64)
    w_excited = -1.0 + E_PARAM
    w_inhibited = -1.0 - D_PARAM
    
    # Cumulative probabilities for fast roulette wheel selection
    p1 = 1.0 - p
    p2 = p1 + p * (1.0 - q) / 2.0
    p3 = p2 + p * (1.0 - q) / 2.0
    
    n = size(W, 1)
    fill!(W, 0.0) # Reset matrix
    
    @inbounds for j in 1:n
        for i in 1:(j-1)
            r = rand()
            if r <= p1
                # Choice -2
                W[i, j] = w_inhibited
                W[j, i] = w_inhibited
            elseif r <= p2
                # Choice -1
                W[i, j] = w_inhibited
                W[j, i] = w_excited
            elseif r <= p3
                # Choice 1
                W[i, j] = w_excited
                W[j, i] = w_inhibited
            else
                # Choice 2
                W[i, j] = w_excited
                W[j, i] = w_excited
            end
        end
    end
    return W
end

# -------------------------------------------------------------------------
# MAIN EXECUTION SCRIPT
# -------------------------------------------------------------------------
function main()
    # Parameters
    E_PARAM = 0.25
    D_PARAM = 0.5
    MATRIX_SIZE = 500
    P_VALS = range(0.0, 1.0, length=11) |> collect
    Q_VALS = range(0.0, 1.0, length=11) |> collect
    NUM_TRIALS = 100
    
    TIME_SPAN = (0.0, 600.0)
    CONVERGENCE_WINDOW = 25
    CONVERGENCE_TOL = 0.01
    ACTIVE_THRESHOLD = 0.05
    DERIV_ERROR_THRESHOLD = 0.01

    num_p = length(P_VALS)
    num_q = length(Q_VALS)

    # Storage arrays
    convergence_heatmap = zeros(num_q, num_p)
    
    # Structuring storage using Matrices of Vectors for fast memory access
    all_final_states = Matrix{Vector{Vector{Float64}}}(undef, num_p, num_q)
    all_weight_matrices = Matrix{Vector{Matrix{Float64}}}(undef, num_p, num_q)
    all_recent_trajectories = Matrix{Vector{Matrix{Float64}}}(undef, num_p, num_q)
    
    for p_idx in 1:num_p, q_idx in 1:num_q
        all_final_states[p_idx, q_idx] = Vector{Vector{Float64}}()
        all_weight_matrices[p_idx, q_idx] = Vector{Matrix{Float64}}()
        all_recent_trajectories[p_idx, q_idx] = Vector{Matrix{Float64}}()
    end

    # Folder setup
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    run_folder = "analysis_run_$(timestamp)_N$(MATRIX_SIZE)"
    mkpath(run_folder)
    
    println("Created analysis folder: ", run_folder)
    println("Starting simulation...")

    # Trajectory Plot Setup (Deferred plotting configuration)
    traj_plots = Any[]

    # Pre-allocate for simulation
    W = zeros(Float64, MATRIX_SIZE, MATRIX_SIZE)
    x0 = zeros(Float64, MATRIX_SIZE)
    prob_base = ODEProblem(system_ode!, x0, TIME_SPAN, W)
    
    for (p_idx, p) in enumerate(P_VALS)
        for (q_idx, q) in enumerate(Q_VALS)
            trial_convergence_results = zeros(Int, NUM_TRIALS)
            
            # For plotting just one sample trajectory per (p, q) configuration
            sample_t = Float64[]
            sample_y = Matrix{Float64}(undef, 0, 0)

            for trial in 1:NUM_TRIALS
                generate_weight_matrix!(W, p, q, E_PARAM, D_PARAM)
                rand!(x0) # Random initial conditions in [0, 1)
                
                # Update problem with new initial conditions and matrix copy
                W_copy = copy(W)
                prob = remake(prob_base, u0=copy(x0), p=W_copy)
                
                # Tsit5 is generally highly efficient for non-stiff or mildly stiff non-linear ODEs
                # Only save the first time step and the last 15 time steps
                save_times = vcat([0.0], collect(range(585.0, 600.0, length=15)))

                sol = solve(prob, Tsit5(), saveat=save_times)
                
                # Extract solution components
                # sol.u is a Vector of Vectors.
                final_state = copy(sol.u[end])
                
                # Extract recent window
                n_steps = length(sol.u)
                recent_idx_start = max(1, n_steps - 14)
                recent_traj = reduce(hcat, sol.u[recent_idx_start:end]) # MATRIX_SIZE x 15
                
                push!(all_final_states[p_idx, q_idx], final_state)
                push!(all_weight_matrices[p_idx, q_idx], W_copy)
                push!(all_recent_trajectories[p_idx, q_idx], recent_traj)

                # Convergence Check
                is_converged = false
                if n_steps >= CONVERGENCE_WINDOW
                    check_start = n_steps - CONVERGENCE_WINDOW + 1
                    max_diff = 0.0
                    @views for step in check_start:n_steps
                        diff = maximum(abs.(final_state .- sol.u[step]))
                        max_diff = max(max_diff, diff)
                    end
                    if max_diff <= CONVERGENCE_TOL
                        is_converged = true
                    end
                end
                trial_convergence_results[trial] = is_converged ? 1 : 0
                
                # Save first trial trajectory for plotting
                if trial == 1
                    sample_t = sol.t
                    sample_y = reduce(hcat, sol.u)
                end
            end
            
            mean_conv = mean(trial_convergence_results)
            convergence_heatmap[q_idx, p_idx] = mean_conv
            @printf("p=%.1f, q=%.1f | Mean Convergence: %.2f\n", p, q, mean_conv)
            
            # Save layout plot
            p_traj = plot(sample_t, sample_y', legend=false, title="p=$(p), q=$(q)", 
                          titlefontsize=6, tickfontsize=4, lw=0.5, grid=false)
            push!(traj_plots, p_traj)
        end
    end
    println("\nSimulation complete. Generating figures...\n")
    
    # Restructure trajectories plot to 11x11 grid layout (Requires large canvas)
    # The order of subplots needs to match Python's arrangement (q ascending upwards)
    ordered_plots = Any[]
    for q_idx in num_q:-1:1
        for p_idx in 1:num_p
            idx = (p_idx - 1) * num_q + q_idx
            push!(ordered_plots, traj_plots[idx])
        end
    end
    fig_trajectories = plot(ordered_plots..., layout=(num_q, num_p), size=(1500, 1500), 
                            plot_title="System Trajectories for Matrix Size $(MATRIX_SIZE)")
    savefig(fig_trajectories, joinpath(run_folder, "system_trajectories.png"))
    
    # -------------------------------------------------------------------------
    # PLOT: Convergence Heatmap
    # -------------------------------------------------------------------------
    q_curve = range(0.0, 1.0, length=100)
    p_curve = (4.0 .* q_curve) ./ ((1.0 .+ q_curve).^2)

    fig_conv = heatmap(P_VALS, Q_VALS, convergence_heatmap, c=:RdYlBu, clims=(0,1),
                       xlabel="p (interaction probability)", ylabel="q (excitatory probability)",
                       title="Convergence Heatmap (N=$(MATRIX_SIZE))", size=(800, 700))
    plot!(fig_conv, p_curve, q_curve, color=:white, lw=2.5, ls=:dash, label="p = 4q/(1+q)^2")
    savefig(fig_conv, joinpath(run_folder, "convergence_heatmap.pdf"))

    # -------------------------------------------------------------------------
    # BLOCK 1: Active Neuron Analysis
    # -------------------------------------------------------------------------
    println("="^70)
    println("BLOCK 1: ACTIVE NEURON ANALYSIS")
    println("="^70)
    
    active_mean_heatmap = zeros(num_q, num_p)
    active_std_heatmap  = zeros(num_q, num_p)
    active_min_heatmap  = zeros(num_q, num_p)
    active_max_heatmap  = zeros(num_q, num_p)

    for p_idx in 1:num_p, q_idx in 1:num_q
        trial_active_counts = [sum(state .> ACTIVE_THRESHOLD) for state in all_final_states[p_idx, q_idx]]
        
        active_mean_heatmap[q_idx, p_idx] = mean(trial_active_counts)
        active_std_heatmap[q_idx, p_idx]  = std(trial_active_counts)
        active_min_heatmap[q_idx, p_idx]  = minimum(trial_active_counts)
        active_max_heatmap[q_idx, p_idx]  = maximum(trial_active_counts)
        
        @printf("p=%.1f, q=%.1f | Active: %.1f±%.1f [%d, %d]\n", 
                P_VALS[p_idx], Q_VALS[q_idx], active_mean_heatmap[q_idx, p_idx], 
                active_std_heatmap[q_idx, p_idx], active_min_heatmap[q_idx, p_idx], active_max_heatmap[q_idx, p_idx])
    end

    expected_max = max(MATRIX_SIZE * 0.8, 3 * log(MATRIX_SIZE))
    actual_max = maximum(active_mean_heatmap)
    vmax_count = min(MATRIX_SIZE, max(expected_max, actual_max * 1.1))
    
    fig_stats_1 = heatmap(P_VALS, Q_VALS, active_mean_heatmap, c=:viridis, clims=(0, vmax_count), title="Mean Active Neurons")
    fig_stats_2 = heatmap(P_VALS, Q_VALS, active_std_heatmap, c=:plasma, clims=(0, vmax_count/4), title="Std Dev of Active Neurons")
    fig_stats_3 = heatmap(P_VALS, Q_VALS, active_min_heatmap, c=:cividis, clims=(0, vmax_count), title="Minimum Active Neurons")
    fig_stats_4 = heatmap(P_VALS, Q_VALS, active_max_heatmap, c=:inferno, clims=(0, vmax_count), title="Maximum Active Neurons")
    
    fig_active_stats = plot(fig_stats_1, fig_stats_2, fig_stats_3, fig_stats_4, layout=(2,2), size=(1000, 900), 
                            plot_title="Active Neuron Statistics (N=$(MATRIX_SIZE))")
    savefig(fig_active_stats, joinpath(run_folder, "active_neurons_statistics.pdf"))

    active_fraction_heatmap = active_mean_heatmap ./ MATRIX_SIZE
    fig_active_frac = heatmap(P_VALS, Q_VALS, active_fraction_heatmap, c=:RdYlGn, clims=(0, 1),
                              title="Fraction of Active Neurons (N=$(MATRIX_SIZE))", size=(800, 700))
    savefig(fig_active_frac, joinpath(run_folder, "active_neurons_fraction.pdf"))


    # -------------------------------------------------------------------------
    # BLOCK 2: Dynamical Regime Classification
    # -------------------------------------------------------------------------
    println("\n", "="^70)
    println("BLOCK 2: DYNAMICAL REGIME CLASSIFICATION")
    println("="^70)
    
    fp_mean_heatmap = fill(NaN, num_q, num_p)
    nl_mean_heatmap = fill(NaN, num_q, num_p)
    
    for p_idx in 1:num_p, q_idx in 1:num_q
        trial_fp_counts = Float64[]
        trial_nl_counts = Float64[]
        
        recent_list = all_recent_trajectories[p_idx, q_idx]
        w_list = all_weight_matrices[p_idx, q_idx]
        
        for (recent_x, W_mat) in zip(recent_list, w_list)
            final_state = recent_x[:, end]
            active_mask = final_state .> ACTIVE_THRESHOLD
            
            # Compute dx = -x + ReLU(Wx + 1) for the recent window
            dx = -recent_x .+ max.(0.0, W_mat * recent_x .+ 1.0)
            
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
        
        if !isempty(trial_fp_counts)
            fp_mean_heatmap[q_idx, p_idx] = mean(trial_fp_counts)
        end
        if !isempty(trial_nl_counts)
            nl_mean_heatmap[q_idx, p_idx] = mean(trial_nl_counts)
        end
        
        fp_str = isempty(trial_fp_counts) ? "NaN" : @sprintf("%.1f", mean(trial_fp_counts))
        nl_str = isempty(trial_nl_counts) ? "NaN" : @sprintf("%.1f", mean(trial_nl_counts))
        @printf("p=%.1f, q=%.1f | Active FP Neurons: %s | Active NL Neurons: %s\n", P_VALS[p_idx], Q_VALS[q_idx], fp_str, nl_str)
    end
    
    max_val = maximum(filter(!isnan, vcat(vec(fp_mean_heatmap), vec(nl_mean_heatmap))))
    
    fig_fp = heatmap(P_VALS, Q_VALS, fp_mean_heatmap, c=:cividis, clims=(0, max_val), title="Fixed Point Realizations")
    plot!(fig_fp, p_curve, q_curve, color=:white, lw=2, ls=:dash, label="p = 4q/(1+q)^2")
    
    fig_nl = heatmap(P_VALS, Q_VALS, nl_mean_heatmap, c=:plasma, clims=(0, max_val), title="Nonlinear Realizations")
    plot!(fig_nl, p_curve, q_curve, color=:white, lw=2, ls=:dash, label="p = 4q/(1+q)^2")
    
    fig_regimes = plot(fig_fp, fig_nl, layout=(1,2), size=(1200, 500), 
                       plot_title="Active Neurons by Dynamical Regime (N=$(MATRIX_SIZE))")
    savefig(fig_regimes, joinpath(run_folder, "dynamical_regimes_heatmap.pdf"))
    
    println("\nFigures saved to: $(run_folder)/")
    println("="^70)
end

# Execute Program
main()