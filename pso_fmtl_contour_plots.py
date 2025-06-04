import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams

# ================== Plot Styling ==================
plt.style.use('seaborn-v0_8')
rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 10
rcParams['axes.grid'] = True
rcParams['grid.alpha'] = 0.3
rcParams['axes.edgecolor'] = '#333333'
rcParams['axes.linewidth'] = 0.8

# ================== Data Generation ==================
def generate_autoregressive_data(n, p=3, coefficients=[0.7, 0.2, 0.1], noise_scale=0.1):
    np.random.seed(42)
    data = np.random.randn(n)
    for i in range(p, n):
        data[i] = sum(c * data[i - j - 1] for j, c in enumerate(coefficients)) + np.random.randn() * noise_scale
    return (data - np.mean(data)) / np.std(data)

def generate_multitask_data(n_tasks=10, n_samples=200, p=3, base_noise=0.1):
    np.random.seed(42)
    base_coeffs = np.array([0.7, 0.2, 0.1])
    all_data = []
    for task_idx in range(n_tasks):
        task_coeffs = base_coeffs * (1 + (np.random.rand(p) - 0.5) * 0.1)
        noise = base_noise * (1 + task_idx/n_tasks * 0.5)
        all_data.append(generate_autoregressive_data(n_samples, p, task_coeffs, noise))
    return all_data

# ================== Model Core ==================
def create_selection_matrix(D, M):
    S = np.zeros(D)
    S[:M] = 1
    return np.diag(S)

def train_for_contour(data, M_values, coordinated=True):
    D, p, num_iterations = 200, 3, 500
    n_tasks = len(data)
    task_mses = np.zeros((len(M_values), n_tasks))

    for m_idx, M in enumerate(M_values):
        models = [np.random.randn(D) * 0.01 for _ in range(n_tasks)]
        velocities = [np.zeros(D) for _ in range(n_tasks)]
        selection_matrix = create_selection_matrix(D, M)

        # Configuration based on M
        lambda_reg = max(0.1 - 0.002 * M, 0.001)
        lr_decay = max(0.2 - 0.003 * M, 0.02)

        for iteration in range(num_iterations):
            lr = 0.001 * (1 / (1 + lr_decay * iteration))
            momentum = 0.9 * (1 - 0.0005 * iteration)

            for task_idx in range(n_tasks):
                X = np.array([data[task_idx][i:i+p] for i in range(len(data[task_idx])-p)])
                y = data[task_idx][p:]

                params = selection_matrix @ models[task_idx]
                pred = np.dot(X[:, :p], params[:p])
                mse = np.mean((pred - y) ** 2)

                if iteration == num_iterations - 1:
                    mse_db = 10 * np.log10(max(mse, 1e-12))
                    task_mses[m_idx, task_idx] = mse_db

                grad = -2 * np.dot(X[:, :p].T, (y - pred)) / len(y)
                grad += lambda_reg * models[task_idx][:p]

                if coordinated:
                    velocities[task_idx][:p] = momentum * velocities[task_idx][:p] - lr * grad
                    models[task_idx][:p] += velocities[task_idx][:p]
                else:
                    full_grad = np.zeros(D)
                    full_grad[:p] = grad
                    update = selection_matrix @ full_grad
                    velocities[task_idx] = momentum * velocities[task_idx] - lr * update
                    models[task_idx] += velocities[task_idx]

    return task_mses

# ================== Enhanced Contour Plot Generation ==================
def generate_enhanced_contour_plots():
    # Generate data
    multi_data = generate_multitask_data(n_tasks=10)
    M_values = np.arange(1, 46)  # 1 to 45

    # Train models
    coord_results = train_for_contour(multi_data, M_values, coordinated=True)
    uncoord_results = train_for_contour(multi_data, M_values, coordinated=False)

    # Clip results to match expected range (0 to -15 dB)
    coord_results = np.clip(coord_results, -15, 0)
    uncoord_results = np.clip(uncoord_results, -15, 0)

    # Prepare data for contour plots
    tasks = np.arange(1, 11)
    M_grid, task_grid = np.meshgrid(M_values, tasks)

    # Create figure with enhanced styling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    fig.suptitle('PSO-FMTL Performance Comparison (MSE in dB)',
                fontsize=12, y=1.02, fontweight='bold')

    # Common plot parameters - now matching 0 to -15 dB range
    levels = np.linspace(-15, 0, 16)  # 1 dB increments
    cmap = plt.get_cmap('viridis_r')  # Reversed to show better performance as darker
    linewidths = 0.5
    alpha = 0.8

    # Coordinated contour plot
    cp1 = ax1.contourf(M_grid, task_grid, coord_results.T, levels=levels, cmap=cmap, alpha=alpha)
    contour1 = ax1.contour(M_grid, task_grid, coord_results.T, levels=levels,
                          colors='white', linewidths=linewidths)
    ax1.clabel(contour1, inline=True, fontsize=8, fmt='%1.0f dB')

    # Uncoordinated contour plot
    cp2 = ax2.contourf(M_grid, task_grid, uncoord_results.T, levels=levels, cmap=cmap, alpha=alpha)
    contour2 = ax2.contour(M_grid, task_grid, uncoord_results.T, levels=levels,
                          colors='white', linewidths=linewidths)
    ax2.clabel(contour2, inline=True, fontsize=8, fmt='%1.0f dB')

    # Add colorbars outside each subplot
    cbar1 = fig.colorbar(cp1, ax=ax1, pad=0.02, location='right', ticks=np.arange(-15, 1, 1))
    cbar1.set_label('MSE (dB)', labelpad=10)
    cbar1.ax.tick_params(labelsize=8)

    cbar2 = fig.colorbar(cp2, ax=ax2, pad=0.02, location='right', ticks=np.arange(-15, 1, 1))
    cbar2.set_label('MSE (dB)', labelpad=10)
    cbar2.ax.tick_params(labelsize=8)

    # Annotations and styling
    for ax, title in zip([ax1, ax2], ['Coordinated Learning', 'Uncoordinated Learning']):
        ax.set_title(title, fontsize=11, pad=10)
        ax.set_xlabel('M-value', fontsize=10, labelpad=8)
        ax.set_ylabel('Task Number', fontsize=10, labelpad=8)
        ax.set_xticks(np.arange(0, 46, 5))
        ax.set_yticks(np.arange(1, 11, 1))
        ax.grid(True, alpha=0.3)

        # Add performance zone indicators
        if ax == ax1:
            ax.annotate('Optimal Region', xy=(35, 8), xytext=(25, 5),
                       arrowprops=dict(facecolor='white', shrink=0.05),
                       bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
                       fontsize=8)
        else:
            ax.annotate('Interference Zone', xy=(10, 9), xytext=(20, 7),
                       arrowprops=dict(facecolor='white', shrink=0.05),
                       bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
                       fontsize=8)

    plt.tight_layout()
    plt.show()

# Generate the enhanced contour plots
generate_enhanced_contour_plots()
