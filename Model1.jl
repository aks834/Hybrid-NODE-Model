import Pkg;
Pkg.add("ModelingToolkit")
Pkg.add("DifferentialEquations")
Pkg.add("Plots")

using ModelingToolkit, DifferentialEquations, Plots

#goal: modify this approach to fit what is seen in the wavelet paper- use 6 different diffeqs find the paper's varitales and initial conditions as well as parameters
#Define diffeqs, D(X) etc for all 6 equations build the system- Simulate through 100 days, compare to data- tune hyper Parameters, 
#then- Implant a Neural Network for one of the variables and see what it generates
#Maybe- after all that is done, compare those two methods to a standard lotka volterra approach

# Define our state variables: state(t) = initial condition
@variables t N(t)=80 E(t)=0 J(t)=0 A(t)=0 D(t)=0 B(t)=0 P(t)=0

# Define our parameters-DONE
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

# Define the differential: takes the derivative with respect to `t`
D = Differential(t)

# Define functions
F_P(N) = rP * N / (kP + N)
F_B(P) = (P^κ) / (kB^κ + P^κ)
R_F(P, A) = F_B(P) * A
R_J(E) = E * exp(-δ * (t - θ))
R_A(J) = J * exp(-δ * (t - τ))

# Define differential equations and algebraic constraints
eqs = [
    D(N) ~ -F_P(N) * P - δ * N + Nin,
    D(E) ~ R_F(P, A) - δ * E,
    D(J) ~ R_J(E) - m * J - δ * J,
    D(A) ~ R_A(J) - m * A - δ * A,
    D(D) ~ -ε * F_B(P) * A - δ * D
]

# Algebraic constraint for B
algebraic_eqs = [
    B ~ β * J + A
]

# Combine into a system
@mtkbuild sys = ODESystem(eqs, t; algebraic_constraints=algebraic_eqs)

# Bring these pieces together into an ODESystem with independent variable t
#@mtkbuild sys = ODESystem(eqs, t)

# Convert from a symbolic to a numerical problem to simulate
tspan = (0.0, 100.0)
prob = ODEProblem(sys, [], tspan)

# Solve the ODE
sol = solve(prob)

# Plot the solution
p1 = plot(sol, vars=(N, E, J, A, D), title="State Variables")
p2 = plot(sol, vars=B, title="Total Predator Density")

plot(p1, p2, layout=(2, 1))