using DifferentialEquations, Plots, Graphs, GraphRecipes, Random, LinearAlgebra


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

mkpath("out")
const N = 10
const X = [cos(2π * i / N) for i in 1:N]
const Y = [sin(2π * i / N) for i in 1:N]

f!(dx, x, W, t) = (mul!(dx, W, x); dx .= -x .+ max.(0.0, dx .+ 1.0))

for p in 0.0:0.1:1.0, q in 0.0:0.1:1.0
    W = zeros(N, N)
    p1, p2, p3 = 1-p, 1-p + p*(1-q)/2, 1-p + p*(1-q)
    for j in 1:N, i in 1:j-1
        r = rand()
        W[i,j], W[j,i] = r <= p1 ? (-1.5, -1.5) : r <= p2 ? (-1.5, -0.75) : r <= p3 ? (-0.75, -1.5) : (-0.75, -0.75)
    end
    
    sol = solve(ODEProblem(f!, rand(N), (0.0, 100.0), W), FBDF())
    savefig(plot(sol, legend=false), "out/traj_p$(round(p,digits=1))_q$(round(q,digits=1)).pdf")
    
    g = SimpleDiGraph(W .> -1.0)
    [rem_edge!(g, i, i) for i in 1:N]
    plt = ne(g) == 0 ? scatter(X, Y, ms=15, c=:lightgray, framestyle=:none, legend=false) : graphplot(g, x=X, y=Y, curves=false)
    savefig(plt, "out/graph_p$(round(p,digits=1))_q$(round(q,digits=1)).pdf")
end