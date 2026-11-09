# PSO-FMTL

Code for **PSO-FMTL: Communication-Efficient Online Federated Multitask Learning via Partial Parameter Sharing**.

Online federated kernel regression with random Fourier features (RFF). Each client exchanges only `M` of `D` model coordinates per round (circularly shifted mask, coordinated or uncoordinated), and the server estimates a task-relation matrix online that the clients use to couple their models.

## Layout
| Folder | Content |
|---|---|
| `code/` | `fed_simulation.py` is the simulator (FedAvg, FedProx, Online FMTL, PSO-Fed, PSO-FMTL). The other scripts run the experiments and make the tables and figures. |
| `results/` | JSON output of every run, logs, and `real_data_output.txt` (HAR and FEMNIST). |
| `figures/` | The PNG figures. |

## Setup
```
pip install -r requirements.txt
cd code
```
Python 3.9+; numpy, scipy and matplotlib are enough for the synthetic experiments. Scripts run in a few minutes on 2 CPU cores, except the 4000-round studies.

## Where each paper item comes from
| Paper item | Script -> output |
|---|---|
| Tables IV, V, VII, Fig. (M and rotation) | `run_all_paper_experiments.py` -> `paper_results_v2.json`; `analyze_paper_results.py`, `make_tables.py`, `make_paper_figures.py` |
| Number of tasks (Table VI, Fig. 4) | `ntasks_experiment.py` -> `ntasks_results.json` |
| Quantized FedAvg | `quant_baseline.py` -> `quant_results.json` |
| beta sweep, smoothing/beta schedules | `beta_sweep.py`, `extra_checks.py` |
| Run times (Sec. III-D) | `timing_check.py` (depends on the machine) |
| Convergence figure (coordinated / uncoordinated) | `MU=0.05 ROUNDS=500 SEEDS=30 SHIFT=M python user_style_figs.py`, then `plot_user_style.py` |
| Horizon / sigma comparison (Sec. VI-C) | `horizon_sigma.py` (`MU=0.05` and `MU=0.2`, 20 seeds) |
| Like-for-like check at M=35 | `consistency_checks.py` |
| Effect of theta | `multitask_theta_figs.py` |
| K=100, 4 clients per round setting of the PSO-Fed paper | `gogineni_setting_comparison.py` |
| Parameter bar graph | `make_params_bar.py` |
| Graphical abstract | `make_graphical_abstract.py` |
| HAR and FEMNIST (Sec. VI-G, Table VIII) | `colab_har_femnist.py` (downloads the data; run in Colab or locally). `real_data_fmtl.py` is the same code with a local-folder interface. Output in `results/real_data_output.txt`. |

`compare_fedavg_psofed_psofmtl.py` is a legacy script that scores the online (per-round) error; its numbers are not comparable with the paper (logs in `results/logs/`).

## Notes
- The metric is held-out test MSE in dB (synthetic) and test accuracy (HAR, FEMNIST).
- The HAR and FEMNIST runs use linear RFF models with unconverged step sizes and no significance tests; see the Limitations section of the paper.
- Add a license file of your choice before publishing.
