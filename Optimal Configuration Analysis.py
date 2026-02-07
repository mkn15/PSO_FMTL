import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

# ================== IEEE Publication Style Setup ==================
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
rcParams['font.size'] = 10
rcParams['axes.titlesize'] = 11
rcParams['axes.labelsize'] = 10
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 8.5
rcParams['figure.titlesize'] = 12
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 600
rcParams['savefig.bbox'] = 'tight'
rcParams['figure.constrained_layout.use'] = True

# ================== OPTIMAL CONFIGURATION ANALYSIS ==================

# Optimal configuration parameters
OPTIMAL_M = 35
COMMUNICATION_REDUCTION = 82.5
FULL_MODEL_SIZE = 200
FULL_COMM_COST = 100

# ================== FIGURE 1: OPTIMAL CONFIGURATION ANALYSIS ==================

# Now we only have 2 subplots instead of 3
fig1, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

# Define consistent colors optimized for IEEE/B&W printing
colors = {
    'PSO-Fed': '#3498db',      # Blue (distinct from red)
    'PSO-FMTL': '#e74c3c',     # Red (distinct from blue)
    'FEMNIST': '#2ecc71',      # Green (medium shade)
    'HAR': '#f39c12',          # Orange (distinct from green)
    'Comm': '#9b59b6',         # Purple (distinct from others)
    'M1': '#1f77b4',           # Light blue (Matplotlib default blue)
    'M10': '#2ca02c',          # Green (Matplotlib default green)
    'M35': '#d62728',          # Red (Matplotlib default red)
}

# 1. Accuracy Comparison at Optimal M=35 (now first subplot)
ax1 = axes[0]
methods = ['PSO-Fed', 'PSO-FMTL']
x_pos2 = np.arange(len(methods))
width = 0.35

# Coordinated mode results
femnist_acc = [0.7735, 0.8750]    # PSO-Fed, PSO-FMTL
har_acc = [0.8927, 0.9500]

bars_fem = ax1.bar(x_pos2 - width/2, femnist_acc, width,
                   color=colors['FEMNIST'],
                   alpha=0.85, edgecolor='black', linewidth=1)
bars_har = ax1.bar(x_pos2 + width/2, har_acc, width,
                   color=colors['HAR'],
                   alpha=0.85, edgecolor='black', linewidth=1)

# Add value labels on bars
for bars, color in [(bars_fem, colors['FEMNIST']), (bars_har, colors['HAR'])]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.003,
                f'{height:.3f}', ha='center', va='bottom',
                fontsize=8.5, fontweight='bold', color=color)

# Add improvement percentages with arrows
improvements = [
    ((femnist_acc[1] - femnist_acc[0]) / femnist_acc[0]) * 100,
    ((har_acc[1] - har_acc[0]) / har_acc[0]) * 100
]

# FEMNIST improvement arrow
y_mid_fem = (femnist_acc[0] + femnist_acc[1]) / 2
ax1.annotate('', xy=(1, femnist_acc[1]), xytext=(1, femnist_acc[0]),
            arrowprops=dict(arrowstyle='<->', color=colors['FEMNIST'], lw=1.5))
ax1.text(1.15, y_mid_fem, f'+{improvements[0]:.1f}%',
         ha='left', va='center', fontsize=9, fontweight='bold',
         color=colors['FEMNIST'])

# HAR improvement arrow
y_mid_har = (har_acc[0] + har_acc[1]) / 2
ax1.annotate('', xy=(1, har_acc[1]), xytext=(1, har_acc[0]),
            arrowprops=dict(arrowstyle='<->', color=colors['HAR'], lw=1.5))
ax1.text(1.15, y_mid_har, f'+{improvements[1]:.1f}%',
         ha='left', va='center', fontsize=9, fontweight='bold',
         color=colors['HAR'])

ax1.set_xlabel('Method', fontsize=10)
ax1.set_ylabel('Test Accuracy', fontsize=10)
ax1.set_title(f'(a) Accuracy Comparison (M={OPTIMAL_M})', fontsize=11, fontweight='bold')
ax1.set_xticks(x_pos2)
ax1.set_xticklabels(methods)
ax1.set_ylim(0.7, 1.0)
ax1.grid(True, alpha=0.15, linestyle=':', axis='y')
ax1.set_axisbelow(True)

# Add legend for datasets
from matplotlib.patches import Patch
legend_elements2 = [
    Patch(facecolor=colors['FEMNIST'], alpha=0.85, edgecolor='black', linewidth=1, label='FEMNIST'),
    Patch(facecolor=colors['HAR'], alpha=0.85, edgecolor='black', linewidth=1, label='UCI HAR')
]
ax1.legend(handles=legend_elements2, loc='upper right', fontsize=8.5, framealpha=0.95)

# 2. Direct Trade-off: Communication vs Accuracy (now second subplot)
ax2 = axes[1]
strategies = ['Full Transmission\n(PSO-Fed M=40)', 'Optimal Configuration\n(PSO-FMTL M=35)']
comm_costs = [FULL_COMM_COST, FULL_COMM_COST * (OPTIMAL_M/FULL_MODEL_SIZE)]
accuracies = [0.830, 0.885]  # FEMNIST coordinated

# Create dual axes
ax2_acc = ax2.twinx()

# Communication bars
bars_comm = ax2.bar(strategies, comm_costs,
                    color=colors['Comm'], alpha=0.85,
                    edgecolor='black', linewidth=1, width=0.5)

# Accuracy line with markers
line_acc, = ax2_acc.plot(strategies, accuracies, 'o-', color=colors['FEMNIST'],
                         linewidth=2.5, markersize=10, markerfacecolor='white',
                         markeredgewidth=2, label='Accuracy')

# Add value labels
for i, (bar, cost) in enumerate(zip(bars_comm, comm_costs)):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1.5,
            f'{cost:.1f} KB', ha='center', va='bottom',
            fontsize=9, fontweight='bold', color=colors['Comm'])

    # Accuracy values
    ax2_acc.text(i, accuracies[i] + 0.003, f'{accuracies[i]:.3f}',
                ha='center', va='bottom', fontsize=9,
                fontweight='bold', color=colors['FEMNIST'])

# Add reduction annotation with arrow
reduction = ((comm_costs[0] - comm_costs[1]) / comm_costs[0]) * 100
ax2.annotate(f'{reduction:.1f}%\nReduction',
             xy=(1, comm_costs[1]), xytext=(0.8, comm_costs[1]/2),
             textcoords='data', ha='center', va='center',
             fontsize=9, color='green', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

# Add accuracy gain annotation
accuracy_gain = ((accuracies[1] - accuracies[0]) / accuracies[0]) * 100
ax2_acc.annotate(f'+{accuracy_gain:.1f}%\nGain',
                xy=(1, accuracies[1]), xytext=(1.2, accuracies[1]),
                textcoords='data', ha='left', va='center',
                fontsize=9, color=colors['FEMNIST'], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=colors['FEMNIST'], lw=1.5))

ax2.set_xlabel('Transmission Strategy', fontsize=10)
ax2.set_ylabel('Communication Cost (KB)', fontsize=10, color=colors['Comm'])
ax2_acc.set_ylabel('FEMNIST Accuracy', fontsize=10, color=colors['FEMNIST'])
ax2.set_title('(b) Direct Trade-off Analysis', fontsize=11, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=colors['Comm'])
ax2_acc.tick_params(axis='y', labelcolor=colors['FEMNIST'])
ax2.set_ylim(0, 110)
ax2_acc.set_ylim(0.82, 0.90)
ax2.grid(True, alpha=0.15, linestyle=':', axis='y')
ax2.set_axisbelow(True)

# Create combined legend
from matplotlib.patches import Patch
legend_elements3 = [
    Patch(facecolor=colors['Comm'], alpha=0.85, label='Communication Cost'),
    line_acc
]
ax2.legend(handles=legend_elements3, loc='upper right',
           fontsize=8.5, framealpha=0.95)

# Add figure subtitle at the bottom
fig1.text(0.5, 0.01, 'Optimal Configuration Analysis: PSO-FMTL with M=35',
          ha='center', fontsize=11, fontweight='bold')

plt.savefig('optimal_configuration_analysis_refined.pdf', bbox_inches='tight', dpi=600)
plt.show()
