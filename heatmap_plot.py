import numpy as np
import matplotlib.pyplot as plt

# Sample RMSE matrices for coordinated and uncoordinated cases
# Replace these with your actual RMSE data
rmse_matrix_coordinated = np.array([
    [0.8, 0.7, 0.6],
    [0.6, 0.5, 0.4],
    [0.4, 0.3, 0.2]
])
rmse_matrix_uncoordinated = np.array([
    [0.9, 0.8, 0.7],
    [0.7, 0.6, 0.5],
    [0.5, 0.4, 0.3]
])

# Define task counts and M values
task_counts = [1, 10, 20]
M_values = [30, 40, 50]  # Adjust M values based on your preference

# Function to plot a contour plot
def plot_contour(rmse_matrix, task_counts, M_values, title, ax):
    X, Y = np.meshgrid(task_counts, M_values)
    Z = rmse_matrix

    # Create a filled contour plot
    contour = ax.contourf(X, Y, Z, cmap='YlGnBu', levels=20)  # Adjust levels for smooth shading
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('RMSE')

    ax.set_title(title)
    ax.set_xlabel('Task Count')
    ax.set_ylabel('M Values')

# Create subplots for contour plots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot the contour plots
plot_contour(rmse_matrix_coordinated, task_counts, M_values, "Contour: RMSE vs Task Count and M Values (Coordinated)", axes[0])
plot_contour(rmse_matrix_uncoordinated, task_counts, M_values, "Contour: RMSE vs Task Count and M Values (Uncoordinated)", axes[1])

# Adjust layout and show the plots
plt.tight_layout()
plt.show()

