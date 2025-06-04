import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d

def generate_smooth_curve(final_rmse, iterations=500, convergence_speed=10):
    x = np.linspace(0, 1, iterations)
    y = final_rmse * (0.3 + 0.7/(1 + np.exp(-convergence_speed*(x-0.3))))
    noise = np.random.normal(0, 0.2, iterations) * (1-x)**3
    return gaussian_filter1d(y + noise, sigma=5)

# Corrected configuration with consistent user group keys
mse_db_values = {
    "PSO-FMTL": {
        1: {"100U": -13, "150U": -13.5, "200U": -14},
        5: {"100U": -14, "150U": -14.5, "200U": -15},
        35: {"100U": -15, "150U": -15.5, "200U": -16}
    },
    "PSO-Fed": {
        1: {"100U": -7, "150U": -7.5, "200U": -8},
        5: {"100U": -8, "150U": -8.5, "200U": -9},
        35: {"100U": -10, "150U": -10.5, "200U": -11}
    }
}

# Maintained your preferred line styles and colors
styles = {
    "PSO-FMTL": {
        "100U": {"color": "#1f77b4", "ls": "-", "marker": "o", "ms": 6, "mew": 1, "markevery": 50},
        "150U": {"color": "#ff7f0e", "ls": "--", "marker": "s", "ms": 5, "mew": 1, "markevery": 50},
        "200U": {"color": "#2ca02c", "ls": ":", "marker": "^", "ms": 5, "mew": 1, "markevery": 50}
    },
    "PSO-Fed": {
        "100U": {"color": "#d62728", "ls": "-", "marker": "x", "ms": 6, "mew": 1.5, "markevery": 40, "alpha": 0.8},
        "150U": {"color": "#9467bd", "ls": "--", "marker": "D", "ms": 5, "mew": 1.5, "markevery": 40, "alpha": 0.8},
        "200U": {"color": "#8c564b", "ls": ":", "marker": "P", "ms": 5, "mew": 1.5, "markevery": 40, "alpha": 0.8}
    }
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
plt.subplots_adjust(wspace=0.25, bottom=0.25)

M_values = [1, 5, 35]  # Focus on these three M values

for idx, M in enumerate(M_values):
    ax = axes[idx]

    # Plot curves with proper dictionary access
    for approach in ["PSO-FMTL", "PSO-Fed"]:
        for user in ["100U", "150U", "200U"]:
            # Access values correctly: mse_db_values[approach][M][user]
            y = generate_smooth_curve(mse_db_values[approach][M][user],
                                    convergence_speed=12 if approach == "PSO-FMTL" else 8)
            ax.plot(y, **styles[approach][user],
                   label=f'{approach.split("-")[1]}-{user}',
                   zorder=3 if approach == "PSO-FMTL" else 2)

    # Single performance delta annotation per user group
    for i, user in enumerate(["100U", "150U", "200U"]):
        delta = abs(mse_db_values["PSO-FMTL"][M][user] - mse_db_values["PSO-Fed"][M][user])
        y_pos = (mse_db_values["PSO-FMTL"][M][user] + mse_db_values["PSO-Fed"][M][user])/2
        ax.annotate(f"Δ{delta:.1f}dB", xy=(480, y_pos), xytext=(5, i*3-4),
                  textcoords='offset points', ha='left', va='center',
                  bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8),
                  fontsize=9)

    ax.set_title(f"M = {M}", fontsize=12, weight='bold', pad=10)
    ax.set_xlabel("Iterations", fontsize=10)
    ax.set_ylabel("MSE (dB)", fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.set_xlim(0, 500)
    ax.set_ylim(-16, -2)  # Adjusted range

# Consolidated legend
legend_elements = [
    Line2D([0], [0], color='#1f77b4', marker='o', linestyle='-', markersize=8, label='FMTL-100U'),
    Line2D([0], [0], color='#ff7f0e', marker='s', linestyle='--', markersize=7, label='FMTL-150U'),
    Line2D([0], [0], color='#2ca02c', marker='^', linestyle=':', markersize=7, label='FMTL-200U'),
    Line2D([0], [0], color='#d62728', marker='x', linestyle='-', markersize=8, label='Fed-100U'),
    Line2D([0], [0], color='#9467bd', marker='D', linestyle='--', markersize=7, label='Fed-150U'),
    Line2D([0], [0], color='#8c564b', marker='P', linestyle=':', markersize=7, label='Fed-200U')
]

fig.legend(handles=legend_elements, loc='lower center',
           ncol=3, bbox_to_anchor=(0.5, 0), fontsize=10, framealpha=1)

# Single performance difference mention
#plt.annotate("PSO-FMTL shows consistent 5-6dB improvement over PSO-Fed across all configurations",
             #xy=(0.5, -0.3), xycoords='axes fraction',
            # ha='center', va='center', fontsize=11, bbox=dict(boxstyle='round', fc='white'))

plt.suptitle("PSO-FMTL vs PSO-Fed Performance Comparison", y=1.02, fontsize=14, weight='bold')
plt.savefig("pso_comparison_refined.png", dpi=150, bbox_inches='tight')
plt.show()
