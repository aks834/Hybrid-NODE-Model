using Pkg

Pkg.add("DifferentialEquations")
Pkg.add("ComponentArrays")
Pkg.add("UnPack")
Pkg.add("DelayDiffEq")
Pkg.add("Plots")
#More "normal" way to input ODE--> not using ModelingToolkit

using DifferentialEquations, ComponentArrays, UnPack, DelayDiffEq, Plots


params = ComponentVector(
  β=5.0,
  δ=0.55,
  Nin=80.0,
  rP=3.3,
  kP=4.3,
  rB=2.25,
  kB=15.0,
  κ=1.25,
  ε=0.25,
  m=0.15,
  θ=0.6,
  τ=1.8,
  a=0.9,
  σ=0.5,
  νA=0.57e-3,
  νP=28e-9
)

u0 = ComponentVector(
 # N=80.0,
  #P=10.0,
  #E=10.0,
  #J=10.0,
  #A=10.0,
  #D=10.0
  N=80000.0,
  P=150.0,
  E=0.5,
  J=0.7,
  A=10.5,
  D=0.5
)

function F_P(h, p, t)
  N, P, E, J, A, D = h(p, t)
  @unpack β, δ, Nin, rP, kP, rB, kB, κ, ε, m, θ, τ, a, σ, νA, νP = p

  rP * N / (kP + N)
end

function F_B(h, p, t)
  N, P, E, J, A, D = h(p, t)
  P = abs(P)
  if (P < 0)
    println("h: ", h(p, t))
  end
  @unpack β, δ, Nin, rP, kP, rB, kB, κ, ε, m, θ, τ, a, σ, νA, νP = p

  rB * (P^κ) / (kB^κ + P^κ)
end

function R_E(h, p, t)
  N, P, E, J, A, D = h(p, t)
  @unpack β, δ, Nin, rP, kP, rB, kB, κ, ε, m, θ, τ, a, σ, νA, νP = p

  F_B(h, p, t) * A
end

function R_J(h, p, t)
  @unpack β, δ, Nin, rP, kP, rB, kB, κ, ε, m, θ, τ, a, σ, νA, νP = p

  (R_E(h, p, t-θ))^(-δ*θ)
end

function R_A(h, p, t)
  @unpack β, δ, Nin, rP, kP, rB, kB, κ, ε, m, θ, τ, a, σ, νA, νP = p

  (R_J(h, p, t-τ))^(-δ*τ)
end

function eqs!(du, u, h, p, t)
  @unpack N, P, E, J, A, D = u
  @unpack β, δ, Nin, rP, kP, rB, kB, κ, ε, m, θ, τ, a, σ, νA, νP = p

  du[1] = δ * Nin - (F_P(h,p,t) * P) - δ * N #Nitrogen
  du[2] = F_P(h,p,t) * P - F_B(h,p,t)*(β*J+A)/ε - δ * P #Algae(prey)
  du[3] = R_E(h,p,t) - R_J(h,p,t) - δ * E #Eggs(predator)
  du[4] = R_J(h,p,t) - R_A(h,p,t) - (m + δ) * J #Juvenile(predator)
  du[5] = β * R_A(h,p,t) - (m + δ) * A #Adult(predator)
  du[6] = m * (J + A) - δ * D #Dead
end

h(p, t) = collect(u0)
tspan = (0.0, 35.0)
outofdomain(u, p, t) = any(u .< -1)

prob = DDEProblem(eqs!, u0, h, tspan, params; constant_lags = [params.θ + params.τ], isoutofdomain=outofdomain)
sol = solve(prob)

#println(sol)
plot(sol)
savefig("values.png")



