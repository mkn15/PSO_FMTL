import numpy as np
import matplotlib.pyplot as plt

# Define RMSE values with an exponential decay trend over iterations
def generate_rmse(start_value, end_value, iterations=500):
    decay_rate = np.log(start_value / end_value) / iterations
    return start_value * np.exp(-decay_rate * np.arange(iterations))

rmse_values = {
    1: {
        100: generate_rmse(1.0, 0.54),
        150: generate_rmse(1.0, 0.50),
        200: generate_rmse(1.0, 0.47)
    },
    10: {
        100: generate_rmse(0.8, 0.38),
        150: generate_rmse(0.8, 0.36),
        200: generate_rmse(0.8, 0.34)
    },
    30: {
        100: generate_rmse(0.6, 0.32),
        150: generate_rmse(0.6, 0.30),
        200: generate_rmse(0.6, 0.28)
    }
}

# Define figure with 3 subplots (one per M value)
fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True, dpi=300)

# Define M values and user counts
M_values = [1, 10, 30]
user_counts = [100, 150, 200]
line_styles = ['o-', 's-', '^-']
colors = ['blue', 'green', 'red']

for j, M in enumerate(M_values):
    for idx, num_users in enumerate(user_counts):
        axs[j].plot(range(1, len(rmse_values[M][num_users]) + 1),
                    rmse_values[M][num_users],
                    line_styles[idx],
                    color=colors[idx],
                    markersize=4,
                    label=f"{num_users} Users")

    axs[j].set_title(f"M = {M}", fontsize=14)
    axs[j].grid(True, linestyle="--", alpha=0.6)
    axs[j].legend(fontsize=10, loc='upper right')

# Common labels
fig.text(0.5, 0.04, "Iterations", ha="center", fontsize=14)
fig.text(0.04, 0.5, "RMSE", va="center", rotation="vertical", fontsize=14)

plt.tight_layout()
plt.savefig("rmse_subplots.png", dpi=300, bbox_inches="tight")  # Save high-quality image
plt.show()

