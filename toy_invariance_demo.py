"""
Toy stochastic collapse: single-neuron sign invariance and two-neuron permutation invariance.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm.auto import tqdm

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "axes.linewidth": 1.0,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.4,
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# Experimental parameters
# ==============================================================================
NUM_REALIZATIONS = 100
NUM_STEPS = 4000
EVAL_INTERVAL = 10
LR = 0.1
LABEL_NOISE_STD = 0.5
DATA_SIZE = 32

torch.manual_seed(42)
X_data = torch.randn(DATA_SIZE, 1).to(device)
Y_true = X_data + 0.1 * torch.randn_like(X_data)

steps_eval = np.arange(0, NUM_STEPS, EVAL_INTERVAL)
num_evals = len(steps_eval)


def gelu(x):
    return torch.nn.functional.gelu(x)


# ==============================================================================
# Experiment 1: single-neuron sign invariance
# ==============================================================================
print("Running Experiment 1: Single Neuron Sign Invariance...")
dist_origin = np.zeros((NUM_REALIZATIONS, num_evals))
traj_exp1 = []

for r in tqdm(range(NUM_REALIZATIONS), desc="Exp 1 Realizations"):
    w1 = torch.nn.Parameter(torch.randn(1, device=device) * 0.8 + 0.5)
    w2 = torch.nn.Parameter(torch.randn(1, device=device) * 0.8 + 0.5)
    optimizer = torch.optim.SGD([w1, w2], lr=LR)

    single_traj = []
    for t in range(NUM_STEPS):
        noise = torch.randn_like(Y_true) * LABEL_NOISE_STD
        Y_noisy = Y_true + noise

        optimizer.zero_grad()
        output = w2 * gelu(w1 * X_data)
        loss = torch.mean((output - Y_noisy) ** 2)
        loss.backward()
        optimizer.step()

        if t % EVAL_INTERVAL == 0:
            idx = t // EVAL_INTERVAL
            d = torch.sqrt(w1 ** 2 + w2 ** 2).item()
            dist_origin[r, idx] = d
            if r == 0:
                single_traj.append([w1.item(), w2.item()])

    if r == 0:
        traj_exp1 = np.array(single_traj)

# ==============================================================================
# Experiment 2: two-neuron permutation invariance
# ==============================================================================
print("Running Experiment 2: Two Neuron Permutation Invariance...")
dist_perm = np.zeros((NUM_REALIZATIONS, num_evals))
traj_exp2 = []

for r in tqdm(range(NUM_REALIZATIONS), desc="Exp 2 Realizations"):
    w1_in = torch.nn.Parameter(torch.randn(2, device=device) * 0.5 + 0.5)
    w2_out = torch.nn.Parameter(torch.randn(2, device=device) * 0.5 + 0.5)
    optimizer = torch.optim.SGD([w1_in, w2_out], lr=LR)

    single_traj = []
    for t in range(NUM_STEPS):
        noise = torch.randn_like(Y_true) * LABEL_NOISE_STD
        Y_noisy = Y_true + noise

        optimizer.zero_grad()
        h = gelu(X_data * w1_in)
        output = torch.sum(h * w2_out, dim=1, keepdim=True)
        loss = torch.mean((output - Y_noisy) ** 2)
        loss.backward()
        optimizer.step()

        if t % EVAL_INTERVAL == 0:
            idx = t // EVAL_INTERVAL
            n1 = torch.stack([w1_in[0], w2_out[0]])
            n2 = torch.stack([w1_in[1], w2_out[1]])
            diff_sq = torch.sum((n1 - n2) ** 2)
            norm_sq = torch.sum(n1 ** 2) + torch.sum(n2 ** 2)
            d = (2.0 * diff_sq / (norm_sq + 1e-8)).item()
            dist_perm[r, idx] = d
            if r == 0:
                single_traj.append([w1_in[0].item(), w1_in[1].item()])

    if r == 0:
        traj_exp2 = np.array(single_traj)

print("Simulations complete. Generating plots...")

# ==============================================================================
# Statistics and plotting
# ==============================================================================
mean_orig = np.mean(dist_origin, axis=0)
var_orig = np.var(dist_origin, axis=0)
rv_orig = var_orig / (mean_orig ** 2 + 1e-8)

mean_perm = np.mean(dist_perm, axis=0)
var_perm = np.var(dist_perm, axis=0)
rv_perm = var_perm / (mean_perm ** 2 + 1e-8)

fig = plt.figure(figsize=(12, 10))
gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.25)

c_main = "#1f77b4"
c_peak = "#d62728"
c_traj1 = "#ff7f0e"
c_traj2 = "#2ca02c"

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(steps_eval, mean_orig, color=c_main, label=r"Ensemble Mean $\langle d(\theta_t, A) \rangle$")
ax1.fill_between(steps_eval, mean_orig - np.sqrt(var_orig), mean_orig + np.sqrt(var_orig), color=c_main, alpha=0.2)
ax1.set_title("(a) Exp 1: Single Neuron Collapse to Sign Invariant Set", pad=8)
ax1.set_ylabel(r"Distance to Origin $||w||_2$")
ax1.set_xlabel("Iteration Step $t$")
ax1.grid(True, linestyle=":")
ax1.legend()

ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(steps_eval, rv_orig, color=c_peak, linewidth=1.5)
ax2.set_title(r"(b) Exp 1: Relative Variance $\mathcal{R}_v(t)$ (Microtransition)", pad=8)
ax2.set_ylabel(r"Relative Variance $\mathcal{R}_v$")
ax2.set_xlabel("Iteration Step $t$")
ax2.grid(True, linestyle=":")

ax3 = fig.add_subplot(gs[0, 1])
ax3.plot(steps_eval, mean_perm, color=c_main, label=r"Ensemble Mean Distance")
ax3.fill_between(steps_eval, mean_perm - np.sqrt(var_perm), mean_perm + np.sqrt(var_perm), color=c_main, alpha=0.2)
ax3.plot(steps_eval, traj_exp2[:, 0] - traj_exp2[:, 1], color=c_traj1, alpha=0.5, linestyle="--",
         label=r"Sample Traj $\Delta w_{in}$")
ax3.set_title("(c) Exp 2: Collapse to Permutation Invariant Set", pad=8)
ax3.set_ylabel(r"Normalized Distance $dist(w_1, w_2)$")
ax3.set_xlabel("Iteration Step $t$")
ax3.grid(True, linestyle=":")
ax3.legend()

ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(steps_eval, rv_perm, color=c_peak, linewidth=1.5)
ax4.set_title(r"(d) Exp 2: Relative Variance $\mathcal{R}_v(t)$ (Microtransition)", pad=8)
ax4.set_ylabel(r"Relative Variance $\mathcal{R}_v$")
ax4.set_xlabel("Iteration Step $t$")
ax4.grid(True, linestyle=":")

plt.suptitle("Stochastic Collapse in Toy Neural Networks", fontsize=14, fontweight="bold", y=0.96)
plt.savefig("neurips_toy_collapse.png", bbox_inches="tight", dpi=300)
plt.show()
