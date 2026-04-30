using DifferentialEquations
using DelimitedFiles
using Random
using Base.Threads

# 1. ODE SYSTEM
function system_ode!(dx, x, W, t)
    n = length(x)
    @inbounds for i in 1:n
        dot_val = zero(eltype(x))
        for j in 1:n
            dot_val += W[i,j] * x[j]
        end
        dx[i] = -x[i] + max(zero(eltype(x)), dot_val + 1.0)
    end
end

# 2. WEIGHT MATRIX GENERATOR
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
                W[i, j] = w_inhibited; W[j, i] = w_inhibited
            elseif r <= p2
                W[i, j] = w_inhibited; W[j, i] = w_excited
            elseif r <= p3
                W[i, j] = w_excited;   W[j, i] = w_inhibited
            else
                W[i, j] = w_excited;   W[j, i] = w_excited
            end
        end
    end
    return W
end

# 3. MULTITHREADED SIMULATION & SINGLE EXPORT
function run_multithreaded_single_export()
    # Parameters
    MATRIX_SIZE = 5
    E_PARAM = 0.25
    D_PARAM = 0.5
    TIME_SPAN = (0.0, 600.0)
    
    P_VALS = range(0.0, 1.0, length=5) 
    Q_VALS = range(0.0, 1.0, length=5)
    num_p = length(P_VALS)
    num_q = length(Q_VALS)
    
    # Thread-safe storage: Array of Matrices
    # Each parameter combination gets its own slot to prevent data races
    all_results = Vector{Matrix{Float64}}(undef, num_p * num_q)
    
    println("Starting parallel simulation on $(Threads.nthreads()) threads...")

    # MULTITHREADED LOOP
    Threads.@threads for p_idx in 1:num_p
        p = P_VALS[p_idx]
        
        # PER-THREAD MEMORY
        W = zeros(Float64, MATRIX_SIZE, MATRIX_SIZE)
        x0 = zeros(Float64, MATRIX_SIZE)
        
        for q_idx in 1:num_q
            q = Q_VALS[q_idx]
            
            generate_weight_matrix!(W, p, q, E_PARAM, D_PARAM)
            rand!(x0)
            
            prob = ODEProblem(system_ode!, x0, TIME_SPAN, W)
            sol = solve(prob, FBDF(), saveat=1.0, reltol=1e-5, abstol=1e-8)
            
            # Create p and q identifier columns
            steps = length(sol.t)
            p_col = fill(p, steps)
            q_col = fill(q, steps)
            
            # Combine: [p, q, t, x1, x2, x3, x4, x5]
            chunk = hcat(p_col, q_col, sol.t, reduce(hcat, sol.u)')
            
            # Calculate flat index and store safely
            linear_idx = (p_idx - 1) * num_q + q_idx
            all_results[linear_idx] = chunk
        end
    end
    
    println("Simulations complete. Compiling and writing to master CSV...")
    
    # Stack all chunks vertically into one massive matrix
    master_data = reduce(vcat, all_results)
    
    # Export
    filename = "master_ctln_data_N$(MATRIX_SIZE).csv"
    writedlm(filename, master_data, ',')
    println("Data successfully saved to: ", filename)
end

run_multithreaded_single_export()