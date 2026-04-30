using CSV
using DataFrames
using Plots

# Use the PyPlot backend if you prefer, or the default GR backend
# pyplot() 

function plot_trajectories(filepath::String, matrix_size::Int)
    println("Loading trajectory data for plotting...")
    df = CSV.read(filepath, DataFrame)
    
    # Extract unique parameter values to determine grid size
    p_vals = sort(unique(df.p))
    q_vals = sort(unique(df.q))
    num_p = length(p_vals)
    num_q = length(q_vals)
    
    ordered_plots = Any[]
    
    # Iterate through q descending so the top of the grid represents high q
    for q in reverse(q_vals)
        for p in p_vals
            # Filter the dataframe for the specific (p, q) pair
            group = subset(df, :p => x -> x .== p, :q => x -> x .== q)
            
            t_data = group.t
            # Extract only the state columns (x1 to xN)
            y_data = Matrix(group[:, 4:(3+matrix_size)]) 
            
            # Create the subplot for this trajectory
            plt = plot(t_data, y_data, legend=false, 
                       title="p=$p, q=$q", titlefontsize=6, 
                       tickfontsize=4, lw=0.5, grid=false)
            push!(ordered_plots, plt)
        end
    end
    
    println("Compiling trajectory grid...")
    fig = plot(ordered_plots..., layout=(num_q, num_p), size=(1200, 1200),
               plot_title="System Trajectories (N=$(matrix_size))")
               
    out_file = "trajectories_grid_N$(matrix_size).png"
    savefig(fig, out_file)
    println("Saved: ", out_file)
end


function plot_statistics(filepath::String, matrix_size::Int)
    println("Loading summary statistics for plotting...")
    df = CSV.read(filepath, DataFrame)
    
    p_vals = sort(unique(df.p))
    q_vals = sort(unique(df.q))
    
    # Initialize matrices for the heatmaps (Rows = q, Columns = p)
    conv_matrix = zeros(length(q_vals), length(p_vals))
    active_matrix = zeros(length(q_vals), length(p_vals))
    
    # Populate the matrices from the flat DataFrame
    for (i, p) in enumerate(p_vals)
        for (j, q) in enumerate(q_vals)
            # Find the exact row for this p and q
            row = subset(df, :p => x -> x .== p, :q => x -> x .== q)[1, :]
            
            conv_matrix[j, i] = row.is_converged
            active_matrix[j, i] = row.active_neurons
        end
    end
    
    # Theoretical boundary curve
    q_curve = range(0.0, 1.0, length=100)
    p_curve = (4.0 .* q_curve) ./ ((1.0 .+ q_curve).^2)
    
    # 1. Convergence Heatmap
    p1 = heatmap(p_vals, q_vals, conv_matrix, c=:RdYlBu, clims=(0,1),
                 xlabel="p", ylabel="q", title="Convergence (1=FP, 0=NL)")
    plot!(p1, p_curve, q_curve, color=:white, lw=2, ls=:dash, label="p = 4q/(1+q)^2")
    
    # 2. Active Neurons Heatmap
    p2 = heatmap(p_vals, q_vals, active_matrix, c=:viridis, clims=(0, matrix_size),
                 xlabel="p", ylabel="q", title="Active Neurons")
    plot!(p2, p_curve, q_curve, color=:white, lw=2, ls=:dash, label="p = 4q/(1+q)^2")
    
    println("Compiling statistics heatmaps...")
    fig = plot(p1, p2, layout=(1,2), size=(1000, 450), 
               plot_title="Network Statistics Summary (N=$(matrix_size))")
               
    out_file = "statistics_heatmaps_N$(matrix_size).png"
    savefig(fig, out_file)
    println("Saved: ", out_file)
end

# Execute the plotting functions
MATRIX_SIZE = 5
plot_trajectories("master_ctln_data_N5.csv", MATRIX_SIZE)
plot_statistics("analysis_summary_N5.csv", MATRIX_SIZE)