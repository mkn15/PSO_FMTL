import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have the RMSE matrices computed
task_counts = [1, 5, 10, 20, 30]
M_values = [1, 10, 30]

# Contour Plot for RMSE Data
def plot_contour(rmse_matrix, title):
    X, Y = np.meshgrid(task_counts, M_values)
    Z = rmse_matrix

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, Z, cmap="viridis", levels=20)
    plt.colorbar(contour, label="RMSE")
    plt.xlabel("Task Count")
    plt.ylabel("M Values")
    plt.title(title)
    plt.show()

# Example RMSE matrices for demonstration
rmse_matrix_coordinated = np.array([
    [-10.3, -9.9, -9.7, -10.1, -10.4],
    [-10.5, -10.3, -10.4, -10.5, -10.6],
    [-10.0, -10.0, -10.0, -10.0, -10.0]
])

rmse_matrix_uncoordinated = np.array([
    [-10.4, -9.9, -9.6, -10.0, -10.3],
    [-10.6, -10.3, -10.4, -10.5, -10.6],
    [-10.0, -10.0, -10.0, -10.0, -10.0]
])

# Plotting contour plots for both cases
plot_contour(rmse_matrix_coordinated, "Contour Plot: RMSE (Coordinated)")
plot_contour(rmse_matrix_uncoordinated, "Contour Plot: RMSE (Uncoordinated)")
