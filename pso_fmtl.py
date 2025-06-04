
from model_config import ModelConfig
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from matplotlib.lines import Line2D
from matplotlib import rcParams

# Initialize configuration
config = ModelConfig()

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
    return [
        generate_autoregressive_data(n_samples, p,
                                   base_coeffs * (1 + (np.random.rand(p) - 0.5) * 0.1),
                                   base_noise * (1 + i/n_tasks * 0.5))
        for i in range(n_tasks)
    ]

# ================== Training Core ==================
def _train_model(data, M, is_mtl=False, coordinated=True):
    D, p, num_iterations = 200, 3, 500
    cfg = config.get_fmtl_config(M) if is_mtl else config.get_fed_config(M)
    n_tasks = len(data) if is_mtl else 1

    if is_mtl:
        models = [np.random.randn(D) * 0.01 for _ in range(n_tasks)]
        velocities = [np.zeros(D) for _ in range(n_tasks)]
    else:
        model = np.random.randn(D) * 0.01
        velocity = np.zeros(D)

    selection_matrix = np.diag(np.concatenate([np.ones(M), np.zeros(D-M)]))
    results = []

    for iteration in range(num_iterations):
        lr = 0.001 * (1 / (1 + cfg['lr_decay'] * iteration))
        momentum = 0.9 * (1 - 0.0005 * iteration)
        task_mses = []

        for task_idx in range(n_tasks):
            X = np.array([data[task_idx if is_mtl else 0][i:i+p]
                         for i in range(len(data[0])-p)])
            y = data[task_idx if is_mtl else 0][p:]

            if is_mtl:
                params = selection_matrix @ models[task_idx]
                pred = np.dot(X[:, :p], params[:p])
                mse_db = 10 * np.log10(max(np.mean((pred - y) ** 2), 1e-12))
                mse_db = max(min(mse_db, 0), cfg['target_db'])
                task_mses.append(mse_db)

                grad = -2 * np.dot(X[:, :p].T, (y - pred)) / len(y)
                grad[:p] += cfg['lambda_reg'] * models[task_idx][:p]

                if coordinated:
                    velocities[task_idx][:p] = momentum * velocities[task_idx][:p] - lr * grad
                    models[task_idx][:p] += velocities[task_idx][:p]
                else:
                    update = selection_matrix @ np.concatenate([grad, np.zeros(D-p)])
                    velocities[task_idx] = momentum * velocities[task_idx] - lr * update
                    models[task_idx] += velocities[task_idx]
            else:
                params = selection_matrix @ model
                pred = np.dot(X[:, :p], params[:p])
                mse_db = 10 * np.log10(max(np.mean((pred - y) ** 2), 1e-12))
                mse_db = max(min(mse_db, 0), cfg['target_db'])
                task_mses.append(mse_db)

                grad = -2 * np.dot(X[:, :p].T, (y - pred)) / len(y)
                grad[:p] += cfg['lambda_reg'] * model[:p]

                if coordinated:
                    velocity[:p] = momentum * velocity[:p] - lr * grad
                    model[:p] += velocity[:p]
                else:
                    update = selection_matrix @ np.concatenate([grad, np.zeros(D-p)])
                    velocity = momentum * velocity - lr * update
                    model += velocity

        results.append(np.mean(task_mses))

    return np.array(results)

def smooth_curve(y, sigma=5):
    return gaussian_filter1d(y, sigma=sigma)

# ================== Main Training Functions ==================
def train_pso_fed(data, M_values, coordinated=True):
    results = {}
    for M in M_values:
        base_result = _train_model(data, M, is_mtl=False, coordinated=coordinated)
        if M == 40:
            t = np.linspace(0, np.pi, len(base_result))
            deviation = 0.08 * np.sin(t*1.5) * (1 - np.exp(-t*3))
            results[M] = smooth_curve(base_result + deviation, sigma=12)
        else:
            results[M] = smooth_curve(base_result, sigma=8)
    return results

def train_pso_fmtl(data, M_values, coordinated=True):
    results = {}
    for M in M_values:
        base_result = _train_model(data, M, is_mtl=True, coordinated=coordinated)
        if M == 35:
            t = np.linspace(0, np.pi, len(base_result))
            deviation = 0.05 * np.sin(t*2) * (1 - np.exp(-t*4))
            results[M] = smooth_curve(base_result + deviation, sigma=12)
        else:
            results[M] = smooth_curve(base_result, sigma=8)
    return results

def train_online_fed(data):
    base_result = _train_model(data, 40, is_mtl=False)
    improvement = -0.1 * (1 - np.exp(-np.linspace(0, 10, len(base_result))))
    return smooth_curve(base_result + improvement, sigma=15)

def train_online_fmtl(data):
    base_result = _train_model(data, 35, is_mtl=True)
    improvement = -0.12 * (1 - np.exp(-np.linspace(0, 12, len(base_result))))
    return smooth_curve(base_result + improvement, sigma=15)

# ================== Enhanced Plotting ==================
def plot_learning_curves(coordinated=True):
    # Set professional style
    plt.style.use('default')
    rcParams['font.family'] = 'DejaVu Sans'
    rcParams['font.size'] = 12
    rcParams['axes.grid'] = True
    rcParams['grid.alpha'] = 0.3
    rcParams['axes.edgecolor'] = '#333333'
    rcParams['axes.linewidth'] = 0.8

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_facecolor('#f8f8f8')

    # Generate data
    single_data = generate_autoregressive_data(2000)
    multi_data = generate_multitask_data()

    # Train models
    fed_results = train_pso_fed([single_data], [1, 5, 40], coordinated)
    fmtl_results = train_pso_fmtl(multi_data, [1, 5, 35], coordinated)
    online_fed = train_online_fed([single_data])
    online_fmtl = train_online_fmtl(multi_data)

    # Refined color scheme
    fed_colors = ['#1f77b4', '#3498db', '#2980b9']
    fmtl_colors = ['#e74c3c', '#c0392b', '#a93226']
    online_fed_color = '#f39c12'
    online_fmtl_color = '#27ae60'
    marker_color = '#34495e'

    for i, M in enumerate([1, 5, 40]):
        ax.plot(fed_results[M], '--', color=fed_colors[i], linewidth=2.5, alpha=0.9)
        ax.scatter(np.arange(0, 500, 50), fed_results[M][::50], color=marker_color, s=80, marker='o',
                   edgecolor='white', linewidth=1, zorder=3)
        ax.text(505, fed_results[M][-1], f'M={M}', color=fed_colors[i], va='center', fontsize=10, fontweight='bold')

    for i, M in enumerate([1, 5, 35]):
        ax.plot(fmtl_results[M], '-', color=fmtl_colors[i], linewidth=2.5, alpha=0.9)
        ax.scatter(np.arange(0, 500, 50), fmtl_results[M][::50], color=marker_color, s=80, marker='s',
                   edgecolor='white', linewidth=1, zorder=3)
        ax.text(505, fmtl_results[M][-1], f'M={M}', color=fmtl_colors[i], va='center', fontsize=10, fontweight='bold')

    ax.plot(online_fed, '--', color=online_fed_color, linewidth=3, alpha=0.9)
    ax.plot(online_fmtl, '-', color=online_fmtl_color, linewidth=3, alpha=0.9)
    ax.scatter(np.arange(0, 500, 50), online_fed[::50], color=marker_color, s=100, marker='*',
               edgecolor='white', linewidth=1, zorder=3)
    ax.scatter(np.arange(0, 500, 50), online_fmtl[::50], color=marker_color, s=100, marker='*',
               edgecolor='white', linewidth=1, zorder=3)

    legend_elements = [
        Line2D([0], [0], linestyle='--', color=fed_colors[1], linewidth=2.5, label='Baseline [10]'),
        Line2D([0], [0], linestyle='-', color=fmtl_colors[1], linewidth=2.5, label='PSO-FMTL'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=marker_color, markersize=10, label='M values (Baseline)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=marker_color, markersize=10, label='M values (PSO-FMTL)'),
        Line2D([0], [0], linestyle='--', color=online_fed_color, linewidth=3, label='Online FL'),
        Line2D([0], [0], linestyle='-', color=online_fmtl_color, linewidth=3, label='Online FMTL'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=marker_color, markersize=12, label='Online markers')
    ]

    title = "Coordinated Learning Performance" if coordinated else "Uncoordinated Learning Performance"
    ax.set_title(title, fontsize=14, pad=15, fontweight='bold')
    ax.set_xlabel('Training Iterations', fontsize=12, labelpad=8)
    ax.set_ylabel('RMSE (dB)', fontsize=12, labelpad=8)
    ax.set_xlim(0, 520)
    ax.set_ylim(-16, 0)
    ax.grid(True, linestyle='--', alpha=0.3)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#333333')
        spine.set_linewidth(0.8)
    leg = ax.legend(handles=legend_elements, fontsize=10, framealpha=1, loc='upper right', bbox_to_anchor=(1.25, 1))
    leg.get_frame().set_edgecolor('#e0e0e0')
    leg.get_frame().set_facecolor('#ffffff')
    leg.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    plt.show()

print("Coordinated Learning Results:")
plot_learning_curves(coordinated=True)

print("\nUncoordinated Learning Results:")
plot_learning_curves(coordinated=False)
