# To run this, you need to have Julia installed and the required packages.
# Open the Julia REPL and run:
# import Pkg
# Pkg.add(["DifferentialEquations", "Plots", "Statistics", "Distributions"])

using DifferentialEquations
using Plots
using Statistics
using Distributions

# ===================================================================
#                          PARAMETERS
# ===================================================================
# Model parameters
const E_PARAM = 0.25  # Epsilon
const D_PARAM = 0.5   # Delta

# Simulation parameters
const MATRIX_SIZE = 100
const P_VALS = 0.0:0.1:1.0
const Q_VALS = 0.0:0.1:1.0
const NUM_TRIALS = 50

# ODE solver and convergence check parameters
const TIME_SPAN = (0.0, 600.0)
const CONVERGENCE_WINDOW = 25   # How many recent time steps to check for convergence
const CONVERGENCE_TOL = 0.01    # Tolerance for checking if values are stable

# ===================================================================
#                         FUNCTIONS
# ===================================================================

function generate_weight_matrix(matrix_size::Int, p::Float64, q::Float64)
    # This function creates the weight matrix. The logic is identical to the Python version.
    probs = [1 - p, p * (1 - q) / 2, p * (1 - q) / 2, p * q]
    dist = Categorical(probs)

    w_excited = -1 + E_PARAM
    w_inhibited = -1 - D_PARAM
    
    mat = zeros(Float64, matrix_size, matrix_size)

    # Iterate over the upper triangle of the matrix (where i < j)
    for j in 2:matrix_size
        for i in 1:(j-1)
            choice_idx = rand(dist)
            
            if choice_idx == 4 # Corresponds to prob p*q -> Symmetric Excitation
                mat[i, j] = w_excited
                mat[j, i] = w_excited
            elseif choice_idx == 1 # Corresponds to prob 1-p -> Symmetric Inhibition
                mat[i, j] = w_inhibited
                mat[j, i] = w_inhibited
            elseif choice_idx == 2 # Corresponds to prob p*(1-q)/2 -> Asymmetric (i inhibits j)
                mat[i, j] = w_inhibited
                mat[j, i] = w_excited
            elseif choice_idx == 3 # Corresponds to prob p*(1-q)/2 -> Asymmetric (i excites j)
                mat[i, j] = w_excited
                mat[j, i] = w_inhibited
            end
        end
    end
    return mat
end

# Define the ODE system in-place for efficiency
function system_ode!(du, u, p, t)
    W = p # The weight matrix is passed as the parameter 'p'
    # --- BUG FIX IS HERE ---
    # The matrix multiplication W*u should not be broadcasted.
    # We use $() to escape it from the @. macro.
    @. du = -u + max(0, $(W * u) + 1)
end

function check_convergence(sol, window::Int, tol::Float64)
    """
    Performs a post-hoc convergence check, mimicking the Python script's logic.
    """
    # Ensure there are enough data points to check
    if length(sol.u) < window
        return false
    end
    
    final_state = sol.u[end]
    # Get the state vectors from the last 'window' time steps
    recent_states = sol.u[end-window+1:end]
    
    # Check if all recent states are within the tolerance of the final state
    for state in recent_states
        # `maximum(abs.(...))` is equivalent to the infinity norm of the difference vector
        if maximum(abs.(final_state - state)) > tol
            return false # If any point is outside the tolerance, it hasn't converged
        end
    end
    
    return true # If all points were within tolerance, it has converged
end


# ===================================================================
#                           MAIN SCRIPT
# ===================================================================

function run_simulation()
    println("Starting Julia simulation (sequential)...")
    
    num_p = length(P_VALS)
    num_q = length(Q_VALS)
    convergence_heatmap = zeros(num_q, num_p)
    
    # Store the solution of the last trial for each (p,q) pair for plotting
    solution_grid = Array{Any}(undef, num_q, num_p)

    # Create a directory for plots if it doesn't exist
    output_dir = "Heatmaps"
    !isdir(output_dir) && mkdir(output_dir)

    # --- Simulation Loop (Sequential) ---
    for p_idx in 1:num_p
        for q_idx in 1:num_q
            p_val = P_VALS[p_idx]
            q_val = Q_VALS[q_idx]
            
            trial_convergence_results = zeros(Int, NUM_TRIALS)
            last_sol = nothing # Variable to hold the last solution for this (p,q) pair

            for trial_num in 1:NUM_TRIALS
                W = generate_weight_matrix(MATRIX_SIZE, p_val, q_val)
                x0 = rand(Float64, MATRIX_SIZE)
                
                prob = ODEProblem(system_ode!, x0, TIME_SPAN, W)
                
                # Solve the ODE. The solver will adaptively choose time steps.
                sol = solve(prob, Tsit5())
                
                last_sol = sol # Keep track of the last solution
                
                # Check for convergence after the simulation is finished
                is_converged = check_convergence(sol, CONVERGENCE_WINDOW, CONVERGENCE_TOL)
                trial_convergence_results[trial_num] = is_converged ? 1 : 0
            end
            
            convergence_heatmap[q_idx, p_idx] = mean(trial_convergence_results)
            solution_grid[q_idx, p_idx] = last_sol # Save the last solution for plotting
            
            println("p=$(round(p_val, digits=1)), q=$(round(q_val, digits=1)) | Mean Convergence: $(round(mean(trial_convergence_results), digits=2))")
        end
    end
    
    println("\nSimulation complete. Generating plots...")

    # --- Visualization: Heatmap ---
    heatmap_plot = heatmap(
        P_VALS, Q_VALS, convergence_heatmap,
        c = :RdYlBu,
        xlabel = "p value",
        ylabel = "q value",
        title = "Convergence Heatmap (Matrix Size: $(MATRIX_SIZE))",
        colorbar_title = "Convergence Probability",
        aspect_ratio = :auto
    )

    # --- Visualization: Trajectories ---
    plot_grid = plot(layout=(num_q, num_p), size=(4000, 4000), legend=false)
    for p_idx in 1:num_p
        for q_idx in 1:num_q
            sol = solution_grid[q_idx, p_idx]
            p_val = P_VALS[p_idx]
            q_val = Q_VALS[q_idx]
            
            # Plot into the correct subplot. Invert q_idx for correct layout (q=0 at bottom).
            plot!(plot_grid[num_q - q_idx + 1, p_idx], sol,
                title="p=$(round(p_val, digits=1)), q=$(round(q_val, digits=1))",
                framestyle=:axes, # Removes top and right spines
                titlefontsize=16 # Match font size from Python script
            )
        end
    end
    plot!(plot_grid, plot_title="System Trajectories for Matrix Size $(MATRIX_SIZE)", plot_titlevspan=0.02, titlefontsize=40)


    # --- Save Figures ---
    # Find a unique filename based on run number
    i = 0
    base_filename = ""
    while true
        base_filename = "MatrixSize_$(MATRIX_SIZE)_Run_$(i)"
        heatmap_path = joinpath(output_dir, "$(base_filename)_heatmap.pdf")
        !isfile(heatmap_path) && break
        i += 1
    end
    
    heatmap_path = joinpath(output_dir, "$(base_filename)_heatmap.pdf")
    trajectories_path = joinpath(output_dir, "$(base_filename)_trajectories.pdf")
    
    savefig(heatmap_plot, heatmap_path)
    println("Heatmap saved to $(heatmap_path)")

    savefig(plot_grid, trajectories_path)
    println("Trajectories plot saved to $(trajectories_path)")
    
    println("\nDone.")
end

# Run the main function
run_simulation()

