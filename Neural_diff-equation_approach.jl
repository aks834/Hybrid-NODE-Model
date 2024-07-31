#I want to apply the optimization method to this program: 
#=implimentation of a hybrid node model based on christopher rackaucas implimentation at: 
https://github.com/ChrisRackauckas/universal_differential_equations/blob/master/LotkaVolterra/hudson_bay.jl =#

import Pkg
# Install packages 
#=
Pkg.add("ModelingToolkit")
Pkg.add("DifferentialEquations")
Pkg.add("Plots")
Pkg.add("OrdinaryDiffEq")
Pkg.add("DataDrivenDiffEq")
Pkg.add("LinearAlgebra")
Pkg.add("Optim")
Pkg.add("Statistics")
Pkg.add("CSV")
Pkg.add("JLD2")
Pkg.add("FileIO")
Pkg.add("Random")
Pkg.add("DataFrames")
Pkg.add("Lux")
Pkg.add("SciMLBase")
Pkg.add("Zygote")
Pkg.add("ComponentArrays")
Pkg.add("Optimization")
Pkg.add("OptimizationOptimJL")
Pkg.add("DelimitedFiles")
Pkg.add("ForwardDiff")=#
Pkg.add("OptimizationOptimisers")
Pkg.add("Optimisers")
Pkg.add("SciMLSensitivity")

using OrdinaryDiffEq
using SciMLSensitivity
using ModelingToolkit
using ForwardDiff
using Zygote
using Optim
using DelimitedFiles
using DataDrivenDiffEq
using LinearAlgebra, ComponentArrays
using Optimisers
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using Lux
using Plots
gr()
using JLD2, FileIO
using Statistics
using Random

# Set a random seed for reproducible behavior
Random.seed!(5443)

svname = "Chemostat"

# Data Preprocessing
chemostat_data = readdlm("/home/aksel/Institute/data/ProcessedData.dat", '\t', Float64, '\n')
Xₙ = Matrix(transpose(chemostat_data[:, 2:3]))
t = chemostat_data[:, 1] .- chemostat_data[1, 1]
xscale = maximum(Xₙ, dims = 2)
Xₙ .= 1f0 ./ xscale .* Xₙ
tspan = (t[1], t[end])

# Plot the data
scatter(t, transpose(Xₙ), xlabel = "t", ylabel = "x(t), y(t)")
plot!(t, transpose(Xₙ), xlabel = "t", ylabel = "x(t), y(t)")

# Gaussian RBF as activation
rbf(x) = exp.(-(x.^2))

# Define the network 2->5->5->5->2
Network = Lux.Chain(
    Lux.Dense(2, 5, rbf),
    Lux.Dense(5, 5, rbf),
    Lux.Dense(5, 5, tanh),
    Lux.Dense(5, 2)
)

# Initialize the parameters for the model
rng = Random.default_rng()
ps, st = Lux.setup(rng, Network)

# Manually initialized parameters for linear birth/decay rates
linear_params = rand(Float64, 2)

# Combine both sets of parameters into a ComponentArray
p = ComponentArray(linear_params=linear_params, ps=ps)

# Define the hybrid model
function ude_dynamics!(du, u, p, t)
    linear_params = p.linear_params
    model_params = p.ps
    û, _ = Network(u, model_params, st)
    du[1] = linear_params[1] * u[1] + û[1]
    du[2] = -linear_params[2] * u[2] + û[2]
end

# Define the problem
prob_nn = ODEProblem(ude_dynamics!, Xₙ[:, 1], tspan, p)

# Define a predictor
function predict(θ, X = Xₙ[:, 1], T = t)
    Array(solve(prob_nn, Vern7(), u0 = X, p = θ,
                tspan = (T[1], T[end]), saveat = T,
                abstol = 1e-6, reltol = 1e-6))
end

# Define parameters for Multiple Shooting
group_size = 5
continuity_term = 200.0f0

function loss(data, pred)
    return sum(abs2, data - pred)
end

function shooting_loss(p)
    return multiple_shoot(p, Xₙ, t, prob_nn, loss, Vern7(),
                          group_size; continuity_term)
end

function loss_neuralode(p)
    X̂ = predict(p)
    sum(abs2, Xₙ - X̂) / size(Xₙ, 2) + convert(eltype(p), 1e-3) * sum(abs2, p.ps) / length(p.ps)
end

# Define a container to track losses
losses = Float64[]

# Define a callback to track progress
function callback(p, l; doplot=false)
    push!(losses, l)
    if length(losses) % 5 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    false  # Continue optimization
end

# Define the optimization function and problem
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, p)

# Use Optimization.jl to solve the problem with Adam optimizer
res1 = Optimization.solve(
    optprob, Optimisers.Adam(0.05); callback = callback, maxiters = 300
)

println("Training loss after $(length(losses)) iterations: $(losses[end])")

# Plot losses
pl_losses = plot(1:length(losses), losses, yaxis = :log10, xlabel = "Iterations", ylabel = "Loss", label = "ADAM (Shooting)", color = :blue)
savefig(pl_losses, joinpath(pwd(), "plots", "$(svname)_losses.pdf"))

# Name the best candidate and retrieve the best candidate
p_trained = res1.minimizer

## Analysis of the trained network
# Interpolate the solution
tsample = t[1]:0.5:t[end]
X̂ = predict(p_trained, Xₙ[:, 1], tsample)

# Trained on noisy data vs real solution
pl_trajectory = scatter(t, transpose(Xₙ), color = :black, label = ["Measurements" nothing], xlabel = "t", ylabel = "x(t), y(t)")
plot!(tsample, transpose(X̂), color = :red, label = ["UDE Approximation" nothing])
savefig(pl_trajectory, joinpath(pwd(), "plots", "$(svname)_trajectory_reconstruction.pdf"))

# Neural network guess
Ŷ = U(X̂, p_trained[3:end])

pl_reconstruction = scatter(tsample, transpose(Ŷ), xlabel = "t", ylabel = "U(x,y)", color = :red, label = ["UDE Approximation" nothing])
plot!(tsample, transpose(Ŷ), color = :red, lw = 2, style = :dash, label = [nothing nothing])
savefig(pl_reconstruction, joinpath(pwd(), "plots", "$(svname)_missingterm_reconstruction.pdf"))
pl_missing = plot(pl_trajectory, pl_reconstruction, layout = (2, 1))
savefig(pl_missing, joinpath(pwd(), "plots", "$(svname)_reconstruction.pdf"))

# Define the recovered, hybrid model with the rescaled dynamics
function recovered_dynamics!(du, u, p, t)
    û = nn_res(u, p[3:end])  # Network prediction
    du[1] = p[1] * u[1] + û[1]
    du[2] = -p[2] * u[2] + û[2]
end

p_model = [p_trained[1:2]; parameters(nn_res)]

estimation_prob = ODEProblem(recovered_dynamics!, Xₙ[:, 1], tspan, p_model)
# Convert for reuse
sys = modelingtoolkitize(estimation_prob)
dudt = ODEFunction(sys)
estimation_prob = ODEProblem(dudt, Xₙ[:, 1], tspan, p_model)
estimate = solve(estimation_prob, Tsit5(), saveat = t)

# Fit the found model
function loss_fit(θ)
    X̂ = Array(solve(estimation_prob, Tsit5(), p = θ, saveat = t))
    sum(abs2, X̂ .- Xₙ)
end

# Post-fit the model
bfgs_optimizer = Optimisers.setup(BFGS(initial_stepnorm = 0.1f0), p_model)
res_fit = Optimization.optimize(
    loss_fit, p_model, bfgs_optimizer,
    OptimizationOptimisers(),
    maxiters = 1000
)
p_fitted = res_fit.minimizer

# Estimate
estimate_rough = solve(estimation_prob, Tsit5(), saveat = 0.1 * mean(diff(t)), p = p_model)
estimate = solve(estimation_prob, Tsit5(), saveat = 0.1 * mean(diff(t)), p = p_fitted)

# Plot
pl_fitted = plot(t, transpose(Xₙ), style = :dash, lw = 2, color = :black, label = ["Measurements" nothing], xlabel = "t", ylabel = "x(t), y(t)")
plot!(estimate_rough, color = :red, label = ["Recovered" nothing])
plot!(estimate, color = :blue, label = ["Recovered + Fitted" nothing])
savefig(pl_fitted, joinpath(pwd(), "plots", "$(svname)recovery_fitting.pdf"))

# Simulation

# Look at long-term prediction
t_long = (0.0f0, 50.0f0)
estimate_long = solve(estimation_prob, Tsit5(), saveat = 0.25f0, tspan = t_long, p = p_fitted)
plot(estimate_long.t, transpose(xscale .* estimate_long[:, :]), xlabel = "t", ylabel = "x(t),y(t)")

# Save the results
save(joinpath(pwd(), "results", "Chemostat_recovery.jld2"),
    "X", Xₙ, "t", t, "neural_network", U, "initial_parameters", p, "trained_parameters", p_trained, # Training
    "losses", losses, "result", nn_res, "recovered_parameters", parameters(nn_res), # Recovery
    "model", recovered_dynamics!, "model_parameter", p_model, "fitted_parameter", p_fitted,
    "long_estimate", estimate_long) # Estimation

# Post Processing and Plots

c1 = 3  # RGBA(174/255,192/255,201/255,1) # Maroon
c2 = :orange  # RGBA(132/255,159/255,173/255,1) # Red
c3 = :blue  # RGBA(255/255,90/255,0,1) # Orange
c4 = :purple  # RGBA(153/255,50/255,204/255,1) # Purple

p3 = scatter(t, transpose(Xₙ), color = [c1 c2], label = ["x data" "y data"],
             title = "Recovered Model from Chemostat Data",
             titlefont = "Helvetica", legendfont = "Helvetica",
             markersize = 5)

plot!(p3, estimate_long, color = [c3 c4], lw = 1, label = ["Estimated x(t)" "Estimated y(t)"])
plot!(p3, [19.99, 20.01], [0.0, maximum(Xₙ) * 1.25], lw = 1, color = :black, label = nothing)
annotate!([(10.0, maximum(Xₙ) * 1.25, text("Training \nData", 12, :center, :top, :black, "Helvetica"))])
savefig(p3, joinpath(pwd(), "plots", "$(svname)full_plot.pdf"))