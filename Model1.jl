import Pkg;
Pkg.add("ModelingToolkit")
Pkg.add("DifferentialEquations")
Pkg.add("Plots")
#Pkg.add("Symbolics")



using ModelingToolkit, DifferentialEquations, Plots


# Define the state variables: state(t) = initial condition
@variables t N(t)=80 E(t)=1 J(t)=1 A(t)=1 D(t)=1 B(t)=1 P(t)=1


#@syms P(t)  # Define P(t) as a symbolic function


# Define parameters
@parameters β=5.0
           δ=0.55
           Nin=80.0
           rP=3.3
           kP=4.3
           rB=2.25
           kB=15.0
           κ=1.25
           ε=0.25
           m=0.15
           θ=0.6
           τ=1.8
           a=0.9
           σ=0.5
           νA=0.57e-3
           νP=28e-9


# Define functions 
F_P(N) = rP * N / (kP + N)
F_B(P) = (rB * (P^κ) )/ (kB^κ + P^κ)


R_E(t) = F_B * (P(t)) * A(t) 
R_J(t) = (R_E * (t-θ))^(-δ*θ)
R_A(t) = (R_J * (t-τ))^(-δ*τ)

# Check the type of the result of these functions
#=
println("Type of F_P(N): ", typeof(F_P(N)))
println("Type of F_B(P): ", typeof(F_B(P)))
println("Type of R_E(t): ", typeof(R_E(t)))
println("Type of R_J(t): ", typeof(R_J(t)))
println("Type of R_A(t): ", typeof(R_A(t)))=#


#Define the differential-take derivative with respect to T
D = Differential(t)

# Define differential equations
eqs = [
   D(N) ~ (δ * Nin) - (F_P(N) * P) - (δ * N)
   D(P) ~ (F_P(N) * P) - (F_B(P)*B/ε) - (δ * P) 
   D(E) ~ R_E - R_J - (δ * E)
   D(J) ~ R_J - R_A - ((m + δ) * J)
   D(A) ~ (β * R_A) - ((m + δ) * A)
   D(D) ~ m(J + A) - (δ * D)
   B ~ β * J + A
]

# Create ODE system without algebraic constraints
@mtkbuild sys = ODESystem(eqs, t)


#Convert from a symbolic problem to a numerical one to simiulate
tspan = (0.0, 10.0)
prob = ODEProblem(sys, [], tspan)


# Solve the ODE
sol = solve(prob)


# Plot the solution
p1 = plot(sol, vars=(N, P, E, J, A, D), title="State Variables")
p2 = plot(sol, vars=B, title="Total Predator Density")


plot(p1, p2, layout=(2, 1))
