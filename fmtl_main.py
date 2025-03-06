import numpy as np
import matplotlib.pyplot as plt

# Function to generate autoregressive data
def generate_autoregressive_data(n, p=3, M=None, variability_factor=0.05):
    np.random.seed(42)
    data = np.random.randn(n)
    noise_scale = 0.1 + (M / 30 if M else 0) * 0.2

    coefficients = [0.7, 0.2, 0.1]
    coefficients = [c + (np.random.rand() - 0.5) * variability_factor for c in coefficients]

    for i in range(p, n):
        data[i] = sum(c * data[i - j - 1] for j, c in enumerate(coefficients)) + np.random.randn() * noise_scale

    data = (data - np.mean(data)) / np.std(data)
    return data

# Function to create the selection matrix
def create_selection_matrix(D, M):
    S = np.zeros(D)
    S[:M] = 1
    return np.diag(S)

# FMTL model with selection and regularization
def fmtl_model_with_selection_and_regularization(num_iterations=500, initial_lr=0.001, p=3, n=2000, D=200, M_values=[1, 10, 30], coordinated=True, lambda_reg=0.1, pre_generated_data=None):
    rmse_db_results = {M: [] for M in M_values}

    for M in M_values:
        print(f"Starting computation for M={M}")

        if pre_generated_data is not None:
            data = pre_generated_data
        else:
            data = generate_autoregressive_data(n, p, M)

        X = np.array([data[i:i + p] for i in range(n - p)])
        y = data[p:n]

        local_model = np.random.randn(D) * 0.01
        selection_matrix = create_selection_matrix(D, M)

        velocity = np.zeros_like(local_model)
        lr = initial_lr

        for iteration in range(num_iterations):
            if M == 10:
                lr = initial_lr * (1 / (1 + 0.1 * iteration))
            elif M == 1:
                lr = initial_lr * (1 / (1 + 0.2 * iteration))
            else:
                lr = initial_lr * (1 / (1 + 0.05 * iteration))

            momentum_factor = 0.9 * (1 - 0.0005 * iteration)

            selected_params = selection_matrix @ local_model
            predictions = np.dot(X[:, :p], selected_params[:p])

            predictions = np.clip(predictions, -1e6, 1e6)
            mse = np.mean((predictions - y) ** 2)
            mse = max(mse, 1e-12)
            rmse = np.sqrt(mse)

            rmse_db = 10 * np.log10(rmse**2 / (np.max(np.abs(y))**2 + 1e-12))
            rmse_db_results[M].append(rmse_db)

            gradient = -2 * np.dot(X[:, :p].T, (y - predictions)) / n
            gradient[:p] += lambda_reg * local_model[:p]

            if coordinated:
                velocity[:p] = momentum_factor * velocity[:p] - lr * gradient
                local_model[:p] += velocity[:p]
            else:
                padded_gradient = np.zeros(D)
                padded_gradient[:p] = gradient
                update = selection_matrix @ padded_gradient
                velocity = momentum_factor * velocity - lr * update
                local_model += velocity

        print(f"Completed computation for M={M}. Final RMSE (dB): {rmse_db_results[M][-1]}")

    return rmse_db_results

# Online FMTL model
def online_fmtl_model(num_iterations=500, initial_lr=0.001, p=3, n=2000, D=200, coordinated=True, lambda_reg=0.1, pre_generated_data=None):
    print(f"Starting computation for Online FMTL model")

    if pre_generated_data is not None:
        data = pre_generated_data
    else:
        data = generate_autoregressive_data(n, p)

    X = np.array([data[i:i + p] for i in range(n - p)])
    y = data[p:n]

    local_model = np.random.randn(D) * 0.01
    velocity = np.zeros_like(local_model)
    lr = initial_lr

    rmse_db_results = []

    for iteration in range(num_iterations):
        lr = initial_lr * (1 / (1 + 0.05 * iteration))
        momentum_factor = 0.9

        predictions = np.dot(X[:, :p], local_model[:p])
        predictions = np.clip(predictions, -1e6, 1e6)
        mse = np.mean((predictions - y) ** 2)
        mse = max(mse, 1e-12)
        rmse = np.sqrt(mse)

        rmse_db = 10 * np.log10(rmse**2 / (np.max(np.abs(y))**2 + 1e-12))
        rmse_db_results.append(rmse_db)

        gradient = -2 * np.dot(X[:, :p].T, (y - predictions)) / n
        gradient += lambda_reg * local_model[:p]

        velocity[:p] = momentum_factor * velocity[:p] - lr * gradient
        local_model[:p] += velocity[:p]

    print(f"Completed computation for Online FMTL model. Final RMSE (dB): {rmse_db_results[-1]}")
    return rmse_db_results

# Running simulations
generated_ar_data = generate_autoregressive_data(2000, p=3)

# Coordinated and Uncoordinated cases
results_with_selection_coordinated = fmtl_model_with_selection_and_regularization(coordinated=True, pre_generated_data=generated_ar_data)
results_online_coordinated = online_fmtl_model(coordinated=True, pre_generated_data=generated_ar_data)
results_with_selection_uncoordinated = fmtl_model_with_selection_and_regularization(coordinated=False, pre_generated_data=generated_ar_data)
results_online_uncoordinated = online_fmtl_model(coordinated=False, pre_generated_data=generated_ar_data)

# Plot results
def plot_comparison_with_online(results_with_selection, results_online, M_values, title, coordinated):
    plt.figure(figsize=(10, 6))
    for M in M_values:
        plt.plot(results_with_selection[M], label=f"PSO-FMTL M={M}")
    plt.plot(results_online, label="Online FMTL", linestyle='--', color='black')
    plt.xlabel('Iteration Index')
    plt.ylabel('RMSE (dB)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

plot_comparison_with_online(results_with_selection_coordinated, results_online_coordinated, [1, 10, 30], "RMSE (dB) Comparison: FMTL vs Online FMTL (Coordinated)", True)
plot_comparison_with_online(results_with_selection_uncoordinated, results_online_uncoordinated, [1, 10, 30], "RMSE (dB) Comparison: FMTL vs Online FMTL (Uncoordinated)", False)
