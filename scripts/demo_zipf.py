#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# Parameters for the Zipf distribution
N = 100  # Number of ranks
s_values = [0, 1, 2, 3, 4, 5, 6, 7]  # Different Zipf parameters to plot

# Generate ranks
ranks = np.arange(1, N + 1)

plt.figure(figsize=(10, 6))

# Plot Zipf distribution for each s value
for s in s_values:
    # Compute Zipf distribution values
    zipf_values = 1 / ranks**s
    # Normalize the distribution to sum to 1
    zipf_values /= np.sum(zipf_values)

    # Plotting each Zipf distribution as a line chart
    plt.plot(ranks, zipf_values, marker="o", linestyle="-", label=f"s = {s}")

    print(s, zipf_values)

# Configure the plot
plt.title("Zipf Distribution with Different Parameters (s)")
plt.xlabel("Rank")
plt.ylabel("Probability")
plt.yscale("log")  # Log scale for better visibility
# plt.xscale('log')  # Log scale for rank as well
plt.grid(True)
plt.legend(title="Zipf Parameter (s)")
plt.show()
