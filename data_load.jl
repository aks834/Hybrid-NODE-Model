using Pkg
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("Plots")

using CSV
using DataFrames
using Plots

# Load the data from CSV
data = CSV.read("C1.csv", DataFrame)

# Accessing columns using the exact names from the CSV file
time_days = data[!, "time (days)"]
algae_cells_ml = data[!, " algae (10^6 cells/ml)"]
rotifers_animals_ml = data[!, " rotifers (animals/ml)"]
egg_ratio = data[!, " egg-ratio"]
eggs_per_ml = data[!, " eggs (per ml)"]
dead_animals_per_ml = data[!, " dead animals (per ml)"]
external_medium_mu_mol_N_l = data[!, " external medium (mu mol N / l)"]

# Plotting
plot(time_days, algae_cells_ml, label="Algae (10^6 cells/ml)", xlabel="Time (days)", ylabel="Value", legend=:topright)
plot!(time_days, rotifers_animals_ml, label="Rotifers (animals/ml)")
plot!(time_days, eggs_per_ml, label="Eggs (per ml)")
plot!(time_days, dead_animals_per_ml, label="Dead animals (per ml)")
plot!(time_days, external_medium_mu_mol_N_l, label="External Medium (mu mol N / l)", linestyle=:dash)

# Save the plot
savefig("output_plot.png")
