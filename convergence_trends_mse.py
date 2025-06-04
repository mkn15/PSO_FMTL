import numpy as np
import matplotlib.pyplot as plt

# Final MSE data from your table (in dB)
data = {
    'M_values': [1, 5, 35],  # Only M=1, M=5, M=35
    'users_100': [-12.5, -13.7, -14.8],
    'users_150': [-13.0, -14.2, -15.3],
    'users_200': [-13.5, -14.7, -15.8]
}

# Create figure for visualization
plt.figure(figsize=(8, 6))

# Convergence Trends
iterations = 500
line_styles = ['-', '--', ':']
colors = ['#4e79a7', '#f28e2b', '#e15759']  # Color for each user group

for i, users in enumerate(['users_100', 'users_150', 'users_200']):
    # Create synthetic convergence curves
    x_vals = np.linspace(0, iterations, 100)
    for j, M in enumerate(data['M_values']):
        final_mse = data[users][j]
        y_vals = final_mse + (5 * np.exp(-x_vals / 100))  # Exponential decay for MSE

        # Set the legend labels for each M and user group combination
        if users == 'users_100':
            label = f'M={M}, 100'
        elif users == 'users_150':
            label = f'M={M}, 150'
        elif users == 'users_200':
            label = f'M={M}, 200'

        # Plot the curve with the appropriate color and line style
        plt.plot(x_vals, y_vals, line_styles[i],
                 color=colors[i],
                 alpha=0.7 if j != 2 else 1,
                 linewidth=1 if j != 2 else 2,
                 label=label)  # Now all M values will appear in the legend

# Set y-axis scale with -8 at bottom, -25 at top
plt.ylim(-16, -7)
plt.gca().invert_yaxis()  # This makes -8 at bottom, -25 at top (lower is better)

# Refine plot (keeping all original styling)
plt.xlabel('Iterations', fontsize=12)
plt.ylabel('MSE (dB)', fontsize=12)
plt.title('Convergence Trends (M=1, 5, 35)', fontsize=14)
plt.axhline(y=-25, color='r', linestyle=':', linewidth=1, alpha=0.7)  # Threshold line
plt.grid(True, linestyle='--', alpha=0.4)

# Add the legend (unchanged)
plt.legend(fontsize=8, framealpha=1)

# Set plot limits (unchanged)
plt.xlim(0, iterations)

# Tight layout for better spacing (unchanged)
plt.tight_layout()

# Save and show the plot (unchanged)
plt.savefig('convergence_trends_mse.png', dpi=300, bbox_inches='tight')
plt.show()
