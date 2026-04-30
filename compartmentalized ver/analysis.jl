using CSV
using DataFrames
using Statistics

function analyze_ctln_data(filepath::String, matrix_size::Int)
    println("Loading raw trajectory data from $filepath...")
    
    # 1. Load Data
    # Since writedlm doesn't output headers, we define them here.
    col_names = [:p, :q, :t]
    append!(col_names, [Symbol("x$i") for i in 1:matrix_size])
    
    # Read into a DataFrame
    df = CSV.read(filepath, DataFrame, header=col_names)

    # 2. Setup Analysis Parameters
    CONVERGENCE_TOL = 0.001 / matrix_size
    ACTIVE_THRESHOLD = 1e-7

    # Prepare a new DataFrame to store the summarized results
    results = DataFrame(
        p = Float64[],
        q = Float64[],
        is_converged = Int[],
        active_neurons = Int[],
        regime = String[]
    )

    println("Grouping data and analyzing trajectories...")
    
    # 3. Analyze by (p, q) Group
    # Grouping ensures we isolate the time series for each specific parameter pair
    for group in groupby(df, [:p, :q])
        p_val = first(group.p)
        q_val = first(group.q)

        # Extract the neuron state columns (x1 to xN) as a Julia Matrix
        state_cols = [Symbol("x$i") for i in 1:matrix_size]
        trajectory = Matrix(group[:, state_cols])' # Transpose to match your previous size: (N, time_steps)

        # Grab the final window (last 20 steps, or all if fewer exist)
        window_size = min(20, size(trajectory, 2))
        recent_traj = trajectory[:, end-window_size+1:end]

        # --- A. CONVERGENCE CHECK ---
        traj_std = vec(std(recent_traj, dims=2))
        is_conv = maximum(traj_std) <= CONVERGENCE_TOL ? 1 : 0

        # --- B. ACTIVE NEURON CHECK ---
        recent_max_state = vec(maximum(recent_traj, dims=2))
        active_count = sum(recent_max_state .> ACTIVE_THRESHOLD)

        # --- C. REGIME CLASSIFICATION ---
        # Without W to calculate the precise derivative error, we use the 
        # trajectory's stability (convergence) to proxy the dynamic regime.
        regime_label = is_conv == 1 ? "Fixed Point" : "Nonlinear"

        # Save the row to our results table
        push!(results, (p_val, q_val, is_conv, active_count, regime_label))
    end

    # 4. Export Summary
    output_file = "analysis_summary_N$(matrix_size).csv"
    CSV.write(output_file, results)
    println("Analysis complete. Summary table saved to: ", output_file)
end

# Ensure you have the packages added: Pkg.add(["CSV", "DataFrames"])
analyze_ctln_data("master_ctln_data_N5.csv", 5)