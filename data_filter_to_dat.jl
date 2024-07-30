using CSV
using DataFrames

# Define the path to the input CSV file and output DAT file
input_file = "/home/aksel/Institute/C1.csv"
output_file = "/home/aksel/Institute/ProcessedData.dat"

# Read the CSV file into a DataFrame
df = CSV.read(input_file, DataFrame)

# Define the column names as symbols
time_col = Symbol("time (days)")
algae_col = Symbol(" algae (10^6 cells/ml)")
rotifers_col = Symbol(" rotifers (animals/ml)")

# Select the required columns
filtered_df = df[:, [time_col, algae_col, rotifers_col]]

# Open the output file for writing
open(output_file, "w") do io
    # Optional(not recommended if running Model2.jl): Write the header
    #println(io, "time\talgae\trotifers")
    
    # Write the data rows
    for row in eachrow(filtered_df)
        println(io, "$(row[time_col])\t$(row[algae_col])\t$(row[rotifers_col])")    
    end
end

