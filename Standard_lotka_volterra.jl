using Pkg; 
#=
Pkg.add("ModelingToolkit")
Pkg.add("DifferentialEquations")
Pkg.add("Plots")=#

using ModelingToolkit, DifferentialEquations, Plots

# Define state variables: state(t) = initial condition
@variables t x(t)=0.44249296 y(t)=4.6280594

# Define our parameters
#alpha = algae growth rate.
#beta = frequency with which predator and prey meet - predation
#gamma = frequency at which predators die 
#delta = population of predators increases proportionally with prey
@parameters α=1.3
            β=0.9
            γ=0.8
            δ=1.8

# Define our differential: Wolves the derivative with respect to `t`
D = Differential(t)

# Define the differential equations
eqs = [D(x) ~ α * x - β * x * y
       D(y) ~ -γ * y + δ * x * y
    ]

# Bring these pieces together into an ODESystem with independent variable t
@mtkbuild sys = ODESystem(eqs, t)

# Convert from a symbolic to a numerical problem to simulate
tspan = (0.0, 50.0)
prob = ODEProblem(sys, [], tspan)

# Solve the ODE
sol = solve(prob)

# Plot the solution
p1 = plot(sol, title = "Prey vs Predators", lw=2, legend=:topright, label=["Prey" "Predators"])
savefig("plots/algae_vs_predators.pdf")

plot(p1)
