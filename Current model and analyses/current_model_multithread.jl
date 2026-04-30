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

# OPTIMIZED ODE SYSTEM

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


# OPTIMIZED WEIGHT MATRIX GENERATOR

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


# MAIN EXECUTION SCRIPT

function main()
    LinearAlgebra.BLAS.set_num_threads(1)

    # Parameters
    E_PARAM = 0.25
    D_PARAM = 0.5
    MATRIX_SIZE = 5
    P_VALS = collect(range(0.0, 1.0, length=11))
    Q_VALS = collect(range(0.0, 1.0, length=11))
    NUM_TRIALS = 1
    
    TIME_SPAN = (0.0, 600.0)
    CONVERGENCE_TOL = 0.001/MATRIX_SIZE
    ACTIVE_THRESHOLD = 1e-7 # Lowered to noise floor to handle O(1/N) steady states
    DERIV_ERROR_THRESHOLD = 1e-3

    num_p = length(P_VALS)
    num_q = length(Q_VALS)

    save_times = vcat([0.0], collect(range(570.0, 600.0, length=15)))

    # Storage Arrays (Array indexing is inherently thread-safe in Julia)
    convergence_heatmap = zeros(num_q, num_p)
    active_mean_heatmap = zeros(num_q, num_p)
    active_std_heatmap  = zeros(num_q, num_p)
    active_min_heatmap  = zeros(num_q, num_p)
    active_max_heatmap  = zeros(num_q, num_p)
    fp_mean_heatmap = fill(NaN, num_q, num_p)
    nl_mean_heatmap = fill(NaN, num_q, num_p)

    # Thread-safe matrix to hold trajectory data
    sample_trajectories = Matrix{Tuple{Vector{Float64}, Matrix{Float64}}}(undef, num_p, num_q)

    # Lock to prevent console text from overlapping
    print_lock = SpinLock()

    # Folder setup
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    run_folder = "analysis_run_$(timestamp)_N$(MATRIX_SIZE)"
    mkpath(run_folder)
    
    println("Created analysis folder: ", run_folder)
    println("Starting parallel simulation on $(Threads.nthreads()) threads...")

    # MULTITHREADED LOOP
    # Distributes the outer p_idx loop across available CPU cores
    Threads.@threads for p_idx in 1:num_p
        p = P_VALS[p_idx]
        
        # PER-THREAD MEMORY: W and x0 MUST be allocated inside the thread
        # to prevent data races and matrix corruption.
        W = zeros(Float64, MATRIX_SIZE, MATRIX_SIZE)
        x0 = zeros(Float64, MATRIX_SIZE)
        prob_base = ODEProblem(system_ode!, x0, TIME_SPAN, W)
        
        for q_idx in 1:num_q
            q = Q_VALS[q_idx]
            
            trial_convergence = zeros(Int, NUM_TRIALS)
            trial_active_counts = zeros(Int, NUM_TRIALS)
            trial_fp_counts = Float64[]
            trial_nl_counts = Float64[]

            for trial in 1:NUM_TRIALS
                generate_weight_matrix!(W, p, q, E_PARAM, D_PARAM)
                rand!(x0) 
                
                W_copy = copy(W)
                prob = remake(prob_base, u0=copy(x0), p=W_copy)
                
                # Auto-switching stiff solver with tightened tolerances
                sol = solve(prob, FBDF(), saveat=save_times, reltol=1e-5, abstol=1e-8)
                print(sol.u)
                
                final_state = sol.u[end]
                
                if length(sol.u) > 1
                    recent_traj = reduce(hcat, sol.u[end-20:end])
                    
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
                
                if trial == 1
                    sample_trajectories[p_idx, q_idx] = (sol.t, reduce(hcat, sol.u)')
                end
            end
            
            # --- AGGREGATE TRIAL STATISTICS ---
            mean_conv = mean(trial_convergence)
            convergence_heatmap[q_idx, p_idx] = mean_conv
            
            active_mean_heatmap[q_idx, p_idx] = mean(trial_active_counts)
            active_std_heatmap[q_idx, p_idx]  = std(trial_active_counts)
            active_min_heatmap[q_idx, p_idx]  = minimum(trial_active_counts)
            active_max_heatmap[q_idx, p_idx]  = maximum(trial_active_counts)
            
            if !isempty(trial_fp_counts)
                fp_mean_heatmap[q_idx, p_idx] = mean(trial_fp_counts)
            end
            if !isempty(trial_nl_counts)
                nl_mean_heatmap[q_idx, p_idx] = mean(trial_nl_counts)
            end
            
            # Lock the console briefly to print the output cleanly
            lock(print_lock) do
                @printf("p=%.1f, q=%.1f | Conv: %.2f | Act: %.1f | FP: %s | NL: %s\n", 
                        p, q, mean_conv, mean(trial_active_counts), 
                        isempty(trial_fp_counts) ? "NaN" : @sprintf("%.1f", mean(trial_fp_counts)),
                        isempty(trial_nl_counts) ? "NaN" : @sprintf("%.1f", mean(trial_nl_counts)))
            end
        end
    end
    println("\nSimulation complete. Generating figures...\n")
    

    # GENERATE PLOTS

    
    # 1. The 11x11 Trajectory Grid
    ordered_plots = Any[]
    for q_idx in num_q:-1:1
        for p_idx in 1:num_p
            t_data, y_data = sample_trajectories[p_idx, q_idx]
            p_traj = plot(t_data, y_data, legend=false, title="p=$(P_VALS[p_idx]), q=$(Q_VALS[q_idx])", 
                          titlefontsize=6, tickfontsize=4, lw=0.5, grid=false)
            push!(ordered_plots, p_traj)
        end
    end
    fig_trajectories = plot(ordered_plots..., layout=(num_q, num_p), size=(1500, 1500), 
                            plot_title="System Trajectories for Matrix Size $(MATRIX_SIZE)")
    savefig(fig_trajectories, joinpath(run_folder, "system_trajectories.png"))
    
    # 2. Convergence Heatmap
    q_curve = range(0.0, 1.0, length=100)
    p_curve = (4.0 .* q_curve) ./ ((1.0 .+ q_curve).^2)

    fig_conv = heatmap(P_VALS, Q_VALS, convergence_heatmap, c=:RdYlBu, clims=(0,1),
                       xlabel="p (connection probability)", ylabel="q (symmetry)",
                       title="Convergence Heatmap (N=$(MATRIX_SIZE))", size=(800, 700))
    plot!(fig_conv, p_curve, q_curve, color=:white, lw=2.5, ls=:dash, label="p = 4q/(1+q)^2")
    savefig(fig_conv, joinpath(run_folder, "convergence_heatmap.pdf"))

# 3. Active Neuron Heatmaps (Pseudo-Log Scale: log10(x + 1))
    
    actual_max = maximum(active_mean_heatmap)
    
    # 1. Transform data using log10(x + 1) to naturally handle zeros 
    # and eliminate any nonsensical negative values.
    log_mean = log10.(active_mean_heatmap .+ 1.0)
    log_std  = log10.(active_std_heatmap .+ 1.0)
    log_min  = log10.(active_min_heatmap .+ 1.0)
    log_max  = log10.(active_max_heatmap .+ 1.0)
    
    # Calculate the upper bound for our color limit
    log_cap = log10(actual_max + 1.0)
    log_cap = max(log_cap, log10(2.0)) # Ensure the scale doesn't crash if the whole matrix is 0
    
    # 2. Create custom, intuitive ticks for the color bar
    # We choose values that look nice on a log scale: 0, 1, 3, 10, 30, 100, etc.
    tick_vals = [0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
    tick_vals = filter(x -> x <= MATRIX_SIZE, tick_vals)
    
    # Ensure the absolute maximum (MATRIX_SIZE) is explicitly labeled at the top
    if !(Float64(MATRIX_SIZE) in tick_vals)
        push!(tick_vals, Float64(MATRIX_SIZE))
    end
    
    # Map the human-readable ticks to their exact physical locations on the logged colorbar
    tick_locs = log10.(tick_vals .+ 1.0)
    tick_labels = string.(round.(Int, tick_vals))
    
    # 3. Plot the heatmaps using the transformed data
    fig_stats_1 = heatmap(P_VALS, Q_VALS, log_mean, c=:viridis, clims=(0.0, log_cap), colorbar_ticks=(tick_locs, tick_labels), title="Mean Active")
    
    # Standard deviation gets a slightly tighter cap for better contrast
    log_std_cap = log10(max(actual_max/3.0, 0.0) + 1.0)
    log_std_cap = max(log_std_cap, log10(1.5))
    fig_stats_2 = heatmap(P_VALS, Q_VALS, log_std, c=:plasma, clims=(0.0, log_std_cap), colorbar_ticks=(tick_locs, tick_labels), title="Std Dev Active")
    
    fig_stats_3 = heatmap(P_VALS, Q_VALS, log_min, c=:cividis, clims=(0.0, log_cap), colorbar_ticks=(tick_locs, tick_labels), title="Min Active")
    
    fig_stats_4 = heatmap(P_VALS, Q_VALS, log_max, c=:inferno, clims=(0.0, log_cap), colorbar_ticks=(tick_locs, tick_labels), title="Max Active")
    
    # 4. Text annotation for items that hit the absolute maximum (using original data)
    for (i, p) in enumerate(P_VALS)
        for (j, q) in enumerate(Q_VALS)
            if active_mean_heatmap[j, i] >= actual_max && actual_max > 0
                mean_str = @sprintf("%.1f", active_mean_heatmap[j, i])
                annotate!(fig_stats_1, p, q, text(mean_str, 7, :white, :center))
            end
            if active_max_heatmap[j, i] >= actual_max && actual_max > 0
                max_str = @sprintf("%.1f", active_max_heatmap[j, i])
                annotate!(fig_stats_4, p, q, text(max_str, 7, :white, :center))
            end
        end
    end
    
    fig_active_stats = plot(fig_stats_1, fig_stats_2, fig_stats_3, fig_stats_4, layout=(2,2), size=(1000, 900), 
                            plot_title="Active Neuron Statistics (N=$(MATRIX_SIZE))")
    savefig(fig_active_stats, joinpath(run_folder, "active_neurons_statistics.pdf"))

    # # 4. Regime Classification Heatmaps
    # max_val_regime = maximum(filter(!isnan, vcat(vec(fp_mean_heatmap), vec(nl_mean_heatmap))))
    
    # fig_fp = heatmap(P_VALS, Q_VALS, fp_mean_heatmap, c=:cividis, clims=(0, max_val_regime), title="Fixed Point Realizations")
    # plot!(fig_fp, p_curve, q_curve, color=:white, lw=2, ls=:dash, label="p = 4q/(1+q)^2")
    
    # fig_nl = heatmap(P_VALS, Q_VALS, nl_mean_heatmap, c=:plasma, clims=(0, max_val_regime), title="Nonlinear Realizations")
    # plot!(fig_nl, p_curve, q_curve, color=:white, lw=2, ls=:dash, label="p = 4q/(1+q)^2")
    
    # fig_regimes = plot(fig_fp, fig_nl, layout=(1,2), size=(1200, 500), 
    #                    plot_title="Active Neurons by Dynamical Regime (N=$(MATRIX_SIZE))")
    # savefig(fig_regimes, joinpath(run_folder, "dynamical_regimes_heatmap.pdf"))
    
    # println("\nFigures saved to: $(run_folder)/")
    # println("="^70)



    # 4. Regime Classification Heatmaps (Selective Log-Scaling)
    
    # FP
    # 1. Transform FP data to log10(x + 1)
    # We use a copy to avoid modifying the original data for future analysis
    log_fp_data = log10.(fp_mean_heatmap .+ 1.0)
    
    # Calculate caps and ticks for FP
    actual_fp_max = maximum(filter(!isnan, fp_mean_heatmap))
    log_fp_cap = log10(actual_fp_max + 1.0)
    
    # Custom ticks for the FP log colorbar
    fp_tick_vals = [0.0, 1.0, 3.0, 10.0, 30.0, 100.0]
    fp_tick_vals = filter(x -> x <= MATRIX_SIZE, fp_tick_vals)
    fp_tick_locs = log10.(fp_tick_vals .+ 1.0)
    fp_tick_labels = string.(round.(Int, fp_tick_vals))

    fig_fp = heatmap(P_VALS, Q_VALS, log_fp_data, 
                     c=:cividis, 
                     clims=(0.0, log_fp_cap), 
                     colorbar_ticks=(fp_tick_locs, fp_tick_labels),
                     title="Fixed Point Realizations (Log Scale)")
    plot!(fig_fp, p_curve, q_curve, color=:white, lw=2, ls=:dash, label="p = 4q/(1+q)^2")
    
    # --- NONLINEAR (Linear-Scaled) ---
    # Per your request, this remains purely linear
    actual_nl_max = maximum(filter(!isnan, nl_mean_heatmap))
    
    fig_nl = heatmap(P_VALS, Q_VALS, nl_mean_heatmap, 
                     c=:plasma, 
                     clims=(0, actual_nl_max), 
                     title="Nonlinear Realizations (Linear Scale)")
    plot!(fig_nl, p_curve, q_curve, color=:white, lw=2, ls=:dash, label="p = 4q/(1+q)^2")
    
    fig_regimes = plot(fig_fp, fig_nl, layout=(1,2), size=(1200, 500), 
                       plot_title="Active Neurons by Dynamical Regime (N=$(MATRIX_SIZE))")
    savefig(fig_regimes, joinpath(run_folder, "dynamical_regimes_heatmap.pdf"))
end

# Execute Program
main()