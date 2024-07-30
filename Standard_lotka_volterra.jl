using Pkg; 
#=
Pkg.add("ModelingToolkit")
Pkg.add("DifferentialEquations")
Pkg.add("Plots")=#

using ModelingToolkit, DifferentialEquations, Plots

#NOW_WHEN I GET HOME--> Use sindy from rackaucas' implimentation to find realistic parameters 
#that fit with the data.-- then do neural network stuff.

# Define state variables: state(t) = initial condition
@variables t x(t)=11 y(t)=5

# Define our parameters
#alpha = algae growth rate.
#beta = frequency with which predator and prey meet - predation
#gamma = frequency at which predators die 
#delta = population of predators increases proportionally with prey
@parameters α=1.5 
            β=1.0 
            γ=3.0 
            δ=1.0

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
p1 = plot(sol, title = "Algae vs Predators(Rotifers)", lw=2, legend=:topright, label=["Algae(10^6 cells/ml)" "Rotifers(animals/ml)"])

plot(p1)
