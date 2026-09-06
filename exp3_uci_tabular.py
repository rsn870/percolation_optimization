"""
UCI tabular experiments with spectral and bootstrap null tests (Appendix F.2, Table 1).
"""
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import load_digits, fetch_openml
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import linregress
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "grid.alpha": 0.4,
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CACHE_DIR = "./uci_datasets_cache"
RESULTS_DIR = "./null_test_results"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ==============================================================================
# Dataset loading, with direct-URL fallback if OpenML is unreachable
# ==============================================================================
def load_uci_dataset(name):
    cache_filepath = os.path.join(CACHE_DIR, f'{name.replace(" ", "_").lower()}.pkl')
    if os.path.exists(cache_filepath):
        print(f"Loading {name} from local cache...")
        return joblib.load(cache_filepath)

    print(f"Fetching {name}...")
    try:
        if name == "Digits":
            X, y = load_digits(return_X_y=True)
            num_classes = 10

        elif name == "German Credit":
            try:
                X, y = fetch_openml(name="credit-g", version=1, return_X_y=True, as_frame=True, parser="auto")
                X = pd.get_dummies(X, drop_first=True).values
                y = LabelEncoder().fit_transform(y)
            except Exception as e:
                print(f"OpenML timeout/error ({e}). Using direct UCI fallback...")
                df = pd.read_csv(
                    "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",
                    sep=" ", header=None)
                X = pd.get_dummies(df.iloc[:, :-1], drop_first=True).values
                y = df.iloc[:, -1].values - 1
            num_classes = 2

        elif name == "Boston Housing":
            try:
                X, y = fetch_openml(data_id=531, return_X_y=True, as_frame=True, parser="auto")
                X = pd.get_dummies(X, drop_first=True).values
                y = y.values
            except Exception as e:
                print(f"OpenML timeout/error ({e}). Using direct Github mirror fallback...")
                df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv")
                X = df.drop("medv", axis=1).values
                y = df["medv"].values
            y = (y > np.median(y)).astype(int)
            num_classes = 2

        elif name == "Heart Disease":
            try:
                X, y = fetch_openml(name="heart-statlog", version=1, return_X_y=True, as_frame=True, parser="auto")
                X = pd.get_dummies(X, drop_first=True).values
                y = LabelEncoder().fit_transform(y)
            except Exception as e:
                print(f"OpenML timeout/error ({e}). Using direct UCI fallback...")
                df = pd.read_csv(
                    "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat",
                    sep=" ", header=None)
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values - 1
            num_classes = 2

        elif name == "Abalone":
            try:
                X, y = fetch_openml(name="abalone", version=1, return_X_y=True, as_frame=True, parser="auto")
                X = pd.get_dummies(X, drop_first=True).values
                y = y.values.astype(float)
            except Exception as e:
                print(f"OpenML timeout/error ({e}). Using direct UCI fallback...")
                df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data",
                                  header=None)
                X = pd.get_dummies(df.iloc[:, :-1], drop_first=True).values
                y = df.iloc[:, -1].values.astype(float)
            y = np.where(y <= 8, 0, np.where(y <= 10, 1, 2))
            num_classes = 3
        else:
            raise ValueError("Unknown dataset")
    except Exception as fatal_e:
        print(f"FATAL ERROR downloading {name}: {fatal_e}")
        return None, None, None

    X_scaled = StandardScaler().fit_transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    joblib.dump((X_tensor, y_tensor, num_classes), cache_filepath)
    print(f"Saved {name} to local cache.")
    return X_tensor, y_tensor, num_classes


# ==============================================================================
# Model and metrics
# ==============================================================================
class UCI_MLP(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 64)
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(64, 32)
        self.act2 = nn.GELU()
        self.out = nn.Linear(32, out_classes)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.out(x)


def compute_spectral_metrics(weight_matrix):
    with torch.no_grad():
        s = torch.linalg.svdvals(weight_matrix)
        p = s / torch.sum(s)
        p = p[p > 1e-9]
        entropy = -torch.sum(p * torch.log(p))
        eff_rank = torch.exp(entropy).item()
        top_s = s[:5].cpu().numpy()
        if len(top_s) < 5:
            top_s = np.pad(top_s, (0, 5 - len(top_s)))
        return eff_rank, top_s


def calculate_dsi(peaks_idx):
    if len(peaks_idx) < 3:
        return None, None
    peak_epochs = peaks_idx + 1
    k = np.arange(1, len(peak_epochs) + 1)
    slope, _, r_val, _, _ = linregress(k, np.log(peak_epochs))
    return np.exp(slope), r_val ** 2


# Shared detrending/detection pipeline, used by both real data and surrogates.
PROMINENCE = 0.05
DISTANCE = 8
LOCAL_SMOOTH_SIGMA = 1.5


def detrend_from_var(var_O, macro_sigma=20.0):
    epochs_arr = np.arange(len(var_O))
    log_var = np.log(var_O + 1e-10)
    slope, intercept, _, _, _ = linregress(epochs_arr, log_var)
    linear_baseline = slope * epochs_arr + intercept
    flat_log_var = log_var - linear_baseline
    macro_flat = gaussian_filter1d(flat_log_var, sigma=macro_sigma, mode="reflect")
    macro_trend_log = macro_flat + linear_baseline
    return log_var - macro_trend_log


def run_detection_pipeline(var_detrended_raw, prominence=PROMINENCE, distance=DISTANCE,
                            local_sigma=LOCAL_SMOOTH_SIGMA):
    Rv_smooth = gaussian_filter1d(var_detrended_raw, sigma=local_sigma)
    denom = np.max(np.abs(Rv_smooth))
    if denom > 0:
        Rv_smooth = Rv_smooth / denom
    peaks_idx, _ = find_peaks(Rv_smooth, prominence=prominence, distance=distance)
    lam, r2 = calculate_dsi(peaks_idx)
    return Rv_smooth, peaks_idx, lam, r2


# ==============================================================================
# Null model 1: phase-randomized (spectral) surrogates
# ==============================================================================
def phase_randomize(x, rng):
    n = len(x)
    f = np.fft.rfft(x)
    amps = np.abs(f)
    phases = rng.uniform(0, 2 * np.pi, size=len(f))
    phases[0] = 0.0
    if n % 2 == 0:
        phases[-1] = 0.0
    f_surr = amps * np.exp(1j * phases)
    return np.fft.irfft(f_surr, n=n)


def spectral_null_test(var_detrended_raw, real_r2, real_lam, n_surrogates=1000, seed=0):
    rng = np.random.default_rng(seed)
    null_r2, null_lam = [], []
    for _ in range(n_surrogates):
        surr = phase_randomize(var_detrended_raw, rng)
        _, _, lam_s, r2_s = run_detection_pipeline(surr)
        if r2_s is not None:
            null_r2.append(r2_s)
            null_lam.append(lam_s)
    null_r2 = np.array(null_r2)
    null_lam = np.array(null_lam)

    out = {
        "n_surrogates_requested": n_surrogates,
        "n_surrogates_with_valid_fit": int(len(null_r2)),
        "real_r2": None if real_r2 is None else float(real_r2),
        "real_lambda": None if real_lam is None else float(real_lam),
    }
    if real_r2 is not None and len(null_r2) > 0:
        p_val = (1 + np.sum(null_r2 >= real_r2)) / (len(null_r2) + 1)
        out["p_value_r2_ge_real"] = float(p_val)
        out["null_r2_mean"] = float(np.mean(null_r2))
        out["null_r2_median"] = float(np.median(null_r2))
        out["null_r2_90th_pct"] = float(np.percentile(null_r2, 90))
        out["null_lambda_mean"] = float(np.mean(null_lam)) if len(null_lam) else None
        out["null_lambda_std"] = float(np.std(null_lam)) if len(null_lam) else None
    out["null_r2_raw"] = null_r2.tolist()
    out["null_lambda_raw"] = null_lam.tolist()
    return out


# ==============================================================================
# Null model 2: seed-resampling bootstrap, recomputes the full pipeline per draw
# ==============================================================================
def seed_bootstrap_test(order_param_matrix, real_r2, real_lam, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    n_real = order_param_matrix.shape[0]
    boot_r2, boot_lam = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n_real, size=n_real)
        resampled = order_param_matrix[idx, :]
        var_O_b = np.var(resampled, axis=0)
        var_detrended_b = detrend_from_var(var_O_b)
        _, _, lam_b, r2_b = run_detection_pipeline(var_detrended_b)
        if r2_b is not None:
            boot_r2.append(r2_b)
            boot_lam.append(lam_b)
    boot_r2, boot_lam = np.array(boot_r2), np.array(boot_lam)

    out = {"n_boot": n_boot, "n_boot_with_valid_fit": int(len(boot_r2))}
    if len(boot_lam) > 0:
        out["lambda_boot_mean"] = float(np.mean(boot_lam))
        out["lambda_boot_std"] = float(np.std(boot_lam))
        out["lambda_boot_ci95"] = [float(np.percentile(boot_lam, 2.5)), float(np.percentile(boot_lam, 97.5))]
        out["r2_boot_mean"] = float(np.mean(boot_r2))
        out["r2_boot_std"] = float(np.std(boot_r2))
    out["boot_r2_raw"] = boot_r2.tolist()
    out["boot_lambda_raw"] = boot_lam.tolist()
    return out


# ==============================================================================
# Experiment runner
# ==============================================================================
DATASETS = ["Digits", "German Credit", "Boston Housing", "Heart Disease", "Abalone"]
NUM_REALIZATIONS = 20
EPOCHS = 100
BATCH_SIZE = 32
LR = 0.05
N_SPECTRAL_SURROGATES = 1000
N_SEED_BOOTSTRAPS = 500

all_results_summary = []

for dataset_name in DATASETS:
    print(f"\n--- Processing {dataset_name} ---")
    X_t, y_t, num_classes = load_uci_dataset(dataset_name)
    if X_t is None:
        continue
    N_SAMPLES, IN_FEATURES = X_t.shape

    order_param_matrix = np.zeros((NUM_REALIZATIONS, EPOCHS))
    loss_matrix = np.zeros((NUM_REALIZATIONS, EPOCHS))
    sval_trajectories = np.zeros((EPOCHS, 5))
    dataset = torch.utils.data.TensorDataset(X_t, y_t)

    for r in tqdm(range(NUM_REALIZATIONS), desc="Realizations"):
        model = UCI_MLP(IN_FEATURES, num_classes).to(device)
        optimizer = optim.SGD(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()
        loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        for epoch in range(EPOCHS):
            model.train()
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            eff_rank, top_s = compute_spectral_metrics(model.fc1.weight)
            order_param_matrix[r, epoch] = eff_rank
            loss_matrix[r, epoch] = epoch_loss / len(loader)
            if r == 0:
                sval_trajectories[epoch, :] = top_s

    mean_O = np.mean(order_param_matrix, axis=0)
    var_O = np.var(order_param_matrix, axis=0)
    mean_loss = np.mean(loss_matrix, axis=0)

    var_detrended_raw = detrend_from_var(var_O)
    Rv_smooth, peaks_idx, lam, r2 = run_detection_pipeline(var_detrended_raw)
    denom = np.max(np.abs(var_detrended_raw))
    var_detrended = var_detrended_raw / denom if denom > 0 else var_detrended_raw

    print(f"Running spectral null ({N_SPECTRAL_SURROGATES} surrogates)...")
    spectral_null = spectral_null_test(var_detrended_raw, r2, lam, n_surrogates=N_SPECTRAL_SURROGATES, seed=42)

    print(f"Running seed-bootstrap null ({N_SEED_BOOTSTRAPS} resamples)...")
    seed_null = seed_bootstrap_test(order_param_matrix, r2, lam, n_boot=N_SEED_BOOTSTRAPS, seed=42)

    result_record = {
        "dataset": dataset_name,
        "n_peaks_detected": int(len(peaks_idx)),
        "lambda_real": None if lam is None else float(lam),
        "r2_real": None if r2 is None else float(r2),
        "spectral_null": {k: v for k, v in spectral_null.items() if k not in ("null_r2_raw", "null_lambda_raw")},
        "seed_bootstrap_null": {k: v for k, v in seed_null.items() if k not in ("boot_r2_raw", "boot_lambda_raw")},
    }
    all_results_summary.append(result_record)

    clean_name = dataset_name.replace(" ", "_").lower()
    with open(os.path.join(RESULTS_DIR, f"{clean_name}_null_tests.json"), "w") as f:
        json.dump({
            "dataset": dataset_name,
            "lambda_real": None if lam is None else float(lam),
            "r2_real": None if r2 is None else float(r2),
            "spectral_null": spectral_null,
            "seed_bootstrap_null": seed_null,
        }, f, indent=2)

    print(f"[{dataset_name}] real lambda={lam}, real R2={r2}")
    if "p_value_r2_ge_real" in spectral_null:
        print(f"[{dataset_name}] spectral-null p(R2>=real) = {spectral_null['p_value_r2_ge_real']:.4f} "
              f"(null R2 median={spectral_null['null_r2_median']:.3f}, "
              f"90th pct={spectral_null['null_r2_90th_pct']:.3f})")
    if "lambda_boot_ci95" in seed_null:
        print(f"[{dataset_name}] seed-bootstrap lambda 95% CI = {seed_null['lambda_boot_ci95']}")

    # --- Plotting: order parameter/loss, detrended cascade, SVD, null histogram ---
    fig = plt.figure(figsize=(22, 5))
    gs = gridspec.GridSpec(1, 4, wspace=0.4)
    epochs_x = np.arange(1, EPOCHS + 1)
    c_main, c_loss, c_peak = "#1f77b4", "#ff7f0e", "#d62728"

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs_x, mean_O, color=c_main, lw=3.0, label=r"Effective Rank $\langle \mathcal{O}(t) \rangle$")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Effective Rank", color=c_main)
    ax1.tick_params(axis="y", labelcolor=c_main)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(epochs_x, mean_loss, color=c_loss, lw=2.0, linestyle="--", label="Train Loss")
    ax1_twin.set_ylabel("Cross Entropy Loss", color=c_loss)
    ax1_twin.tick_params(axis="y", labelcolor=c_loss)
    ax1.set_title(f"(a) {dataset_name}: Order Parameter & Loss (N={NUM_REALIZATIONS} seeds)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="upper right")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs_x, var_detrended, color="#8c564b", alpha=0.3, lw=1.0, label="Raw Fluctuations")
    ax2.plot(epochs_x, Rv_smooth, color="#8c564b", lw=2.5, label=r"Smoothed $\mathcal{R}_v(t)$")
    for i, idx in enumerate(peaks_idx):
        tp = epochs_x[idx]
        ax2.scatter(tp, Rv_smooth[idx], color=c_peak, s=90, zorder=5)
        ax2.annotate(f"$p_{i+1}$", (tp, Rv_smooth[idx]), textcoords="offset points",
                     xytext=(0, 10), ha="center", color=c_peak, fontweight="bold")
    dsi_title = f"(b) {dataset_name}: Microtransition Variance Peaks"
    if lam is not None:
        dsi_title += f"\n(DSI $\\lambda={lam:.2f}, R^2={r2:.2f}$)"
    ax2.set_title(dsi_title)
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Detrended Log-Variance Fluctuation")
    ax2.set_ylim(-1.2, 1.4)
    ax2.grid(True, linestyle=":")
    ax2.legend(loc="upper right")

    ax3 = fig.add_subplot(gs[0, 2])
    colors_c = plt.cm.viridis(np.linspace(0, 0.9, 5))
    for k in range(5):
        ax3.plot(epochs_x, sval_trajectories[:, k], color=colors_c[k], lw=2.0, label=f"Singular Value {k+1}")
    ax3.set_title(f"(c) {dataset_name}: Dimensionality Collapse (SVD)")
    ax3.set_xlabel("Epochs")
    ax3.set_ylabel(r"Singular Value Magnitude $\sigma_k$")
    ax3.grid(True, linestyle=":")
    ax3.legend(loc="upper left")

    ax4 = fig.add_subplot(gs[0, 3])
    if len(spectral_null["null_r2_raw"]) > 0:
        ax4.hist(spectral_null["null_r2_raw"], bins=30, color="#7f7f7f", alpha=0.7,
                 label=f"Phase-randomized null\n(n={len(spectral_null['null_r2_raw'])})")
        if r2 is not None:
            ax4.axvline(r2, color=c_peak, lw=2.5, linestyle="--", label=f"Real $R^2$={r2:.2f}")
            p_txt = spectral_null.get("p_value_r2_ge_real", None)
            if p_txt is not None:
                ax4.text(0.02, 0.95, f"p = {p_txt:.3f}", transform=ax4.transAxes, va="top", ha="left",
                         fontsize=10, bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax4.set_title(f"(d) {dataset_name}: Null-Model Control\n(spectral surrogate $R^2$ distribution)")
    ax4.set_xlabel(r"Fitted $R^2$ (log-linear DSI fit)")
    ax4.set_ylabel("Count (surrogates)")
    ax4.grid(True, linestyle=":", alpha=0.5)
    ax4.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"Experiment 3: SGD Topological Dynamics on {dataset_name}", y=1.03, fontsize=15, fontweight="bold")
    plt.savefig(f"exp3_uci_{clean_name}_detrended_with_null.png", bbox_inches="tight", dpi=300)
    plt.close()

with open(os.path.join(RESULTS_DIR, "summary_all_datasets.json"), "w") as f:
    json.dump(all_results_summary, f, indent=2)

print("\n=== Null-Model Test Summary (all datasets) ===")
for rec in all_results_summary:
    sn = rec["spectral_null"]
    sb = rec["seed_bootstrap_null"]
    print(f"{rec['dataset']:15s} | peaks={rec['n_peaks_detected']} | "
          f"lambda={rec['lambda_real']} | R2={rec['r2_real']} | "
          f"p(spectral)={sn.get('p_value_r2_ge_real')} | "
          f"lambda 95% CI (seed-boot)={sb.get('lambda_boot_ci95')}")

print(f"\nFull per-dataset JSON results written to: {RESULTS_DIR}/")
print("All datasets processed.")
