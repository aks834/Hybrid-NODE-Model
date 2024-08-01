using Pkg; 
Pkg.add("OrdinaryDiffEq")
Pkg.add("ModelingToolkit")
Pkg.add("DataDrivenDiffEq")
Pkg.add("LinearAlgebra")
Pkg.add("ComponentArrays")
Pkg.add("Optimization")
Pkg.add("OptimizationOptimisers")
Pkg.add("OptimizationOptimJL")
#Pkg.add("DiffEqSensitivity")
Pkg.add("Lux")
Pkg.add("Plots")
Pkg.add("JLD2")
Pkg.add("FileIO")
Pkg.add("Statistics")


using OrdinaryDiffEq
using ModelingToolkit
using DataDrivenDiffEq
using LinearAlgebra, ComponentArrays
using Optimization, OptimizationOptimisers, OptimizationOptimJL #OptimizationFlux for ADAM and OptimizationOptimJL for BFGS
#using DiffEqSensitivity
using Lux
using Plots
gr()
using JLD2, FileIO
using Statistics
# Set a random seed for reproduceable behaviour
using Random
rng = Random.default_rng()
Random.seed!(1234)

#### NOTE
# Since the recent release of DataDrivenDiffEq v0.6.0 where a complete overhaul of the optimizers took
# place, SR3 has been used. Right now, STLSQ performs better and has been changed.

# Create a name for saving ( basically a prefix )
svname = "Ideal_Data_"

## Data generation
function lotka!(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α*u[1] - β*u[2]*u[1]
    du[2] = γ*u[1]*u[2]  - δ*u[2]
end

# Define the experimental parameter
tspan = (0.0,3.0)
u0 = [0.44249296,4.S

## Training

# First train with ADAM for better convergence -> move the parameters into a
# favourable starting positing for BFGS
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p)->loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p))
res1 = Optimization.solve(optprob, ADAM(0.1), callback=callback, maxiters = 200)
println("Training loss after $(length(losses)) iterations: $(losses[end])")
# Train with BFGS
optprob2 = Optimization.OptimizationProblem(optf, res1.minimizer)
res2 = Optimization.solve(optprob2, Optim.BFGS(initial_stepnorm=0.01), callback=callback, maxiters = 10000)
println("Final training loss after $(length(losses)) iterations: $(losses[end])")

# Plot the losses
pl_losses = plot(1:200, losses[1:200], yaxis = :log10, xaxis = :log10, xlabel = "Iterations", ylabel = "Loss", label = "ADAM", color = :blue)
plot!(201:length(losses), losses[201:end], yaxis = :log10, xaxis = :log10, xlabel = "Iterations", ylabel = "Loss", label = "BFGS", color = :red)
savefig(pl_losses, joinpath(pwd(), "plots", "$(svname)_losses.pdf"))
# Rename the best candidate
p_trained = res2.minimizer

## Analysis of the trained network
# Plot the data and the approximation
ts = first(solution.t):mean(diff(solution.t))/2:last(solution.t)
X̂ = predict(p_trained, Xₙ[:,1], ts)
# Trained on noisy data vs real solution
pl_trajectory = plot(ts, transpose(X̂), xlabel = "t", ylabel ="x(t), y(t)", color = :red, label = ["UDE Approximation" nothing])
scatter!(solution.t, transpose(Xₙ), color = :black, label = ["Measurements" nothing])
savefig(pl_trajectory, joinpath(pwd(), "plots", "$(svname)_trajectory_reconstruction.pdf"))

# Ideal unknown interactions of the predictor
Ȳ = [-p_[2]*(X̂[1,:].*X̂[2,:])';p_[3]*(X̂[1,:].*X̂[2,:])']
# Neural network guess
Ŷ = U(X̂,p_trained,st)[1]

pl_reconstruction = plot(ts, transpose(Ŷ), xlabel = "t", ylabel ="U(x,y)", color = :red, label = ["UDE Approximation" nothing])
plot!(ts, transpose(Ȳ), color = :black, label = ["True Interaction" nothing])
savefig(pl_reconstruction, joinpath(pwd(), "plots", "$(svname)_missingterm_reconstruction.pdf"))

# Plot the error
pl_reconstruction_error = plot(ts, norm.(eachcol(Ȳ-Ŷ)), yaxis = :log, xlabel = "t", ylabel = "L2-Error", label = nothing, color = :red)
pl_missing = plot(pl_reconstruction, pl_reconstruction_error, layout = (2,1))
savefig(pl_missing, joinpath(pwd(), "plots", "$(svname)_missingterm_reconstruction_and_error.pdf"))
pl_overall = plot(pl_trajectory, pl_missing)
savefig(pl_overall, joinpath(pwd(), "plots", "$(svname)_reconstruction.pdf"))
