# inverse_pinn_spin_estimation.py

import os
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ============================================================
# Reproducibility and device
# ============================================================

torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTDIR = "inverse_pinn_spin_results"
os.makedirs(OUTDIR, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# PINN network
# ============================================================

class MLP(nn.Module):
    def __init__(self, layers=(1, 64, 64, 64, 64, 1)):
        super().__init__()
        modules = []
        for i in range(len(layers) - 2):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            modules.append(nn.Tanh())
        modules.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*modules)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, theta):
        return self.net(theta)


# ============================================================
# Reduced scalar angular Teukolsky-like residual
# ============================================================

def angular_residual(model, theta, a, m=0.0, omega=0.5, lam=2.0, eps=1e-6):
    """
    Residual:
    d/dθ( sinθ dψ/dθ )
    + [ λ - m^2/sin^2θ - a^2 ω^2 cos^2θ ] ψ = 0
    """

    theta = theta.clone().detach().requires_grad_(True)
    psi = model(theta)

    dpsi = torch.autograd.grad(
        psi,
        theta,
        grad_outputs=torch.ones_like(psi),
        create_graph=True
    )[0]

    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)

    flux = sin_theta * dpsi

    dflux = torch.autograd.grad(
        flux,
        theta,
        grad_outputs=torch.ones_like(flux),
        create_graph=True
    )[0]

    safe_sin2 = sin_theta**2 + eps

    potential = lam - (m**2 / safe_sin2) - (a**2) * (omega**2) * (cos_theta**2)

    residual = dflux + potential * psi
    return residual


def bounded_spin(raw_a, a_min=0.0, a_max=0.99):
    """
    Maps unconstrained raw parameter to physical spin range [0, 0.99].
    """
    return a_min + (a_max - a_min) * torch.sigmoid(raw_a)


# ============================================================
# Forward PINN: generate clean synthetic reference solution
# ============================================================

def train_forward_reference(
    a_true,
    seed=0,
    epochs=8000,
    lr=1e-3,
    n_collocation=300,
    m=0.0,
    omega=0.5,
    lam=2.0,
    verbose=False
):
    """
    Trains a forward PINN for a fixed a_true.
    This generates a clean normalized synthetic angular-mode profile.
    """

    set_seed(seed)

    model = MLP().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    theta_f = torch.linspace(1e-4, math.pi - 1e-4, n_collocation, device=DEVICE).view(-1, 1)
    theta_0 = torch.tensor([[1e-4]], device=DEVICE)
    theta_pi = torch.tensor([[math.pi - 1e-4]], device=DEVICE)
    theta_mid = torch.tensor([[math.pi / 2]], device=DEVICE)

    a_tensor = torch.tensor(a_true, device=DEVICE)

    for epoch in range(epochs):
        optimizer.zero_grad()

        res = angular_residual(model, theta_f, a_tensor, m=m, omega=omega, lam=lam)
        loss_pde = torch.mean(res**2)

        loss_bc = model(theta_0).pow(2).mean() + model(theta_pi).pow(2).mean()

        # Avoid trivial zero solution
        loss_norm = (model(theta_mid) - 1.0).pow(2).mean()

        loss = loss_pde + 10.0 * loss_bc + 10.0 * loss_norm
        loss.backward()
        optimizer.step()

        if verbose and epoch % 1000 == 0:
            print(
                f"[FORWARD] a={a_true:.2f}, epoch={epoch}, "
                f"loss={loss.item():.3e}, pde={loss_pde.item():.3e}"
            )

    return model


# ============================================================
# Inverse PINN: infer spin parameter a from noisy observations
# ============================================================

def train_inverse_pinn(
    theta_data,
    psi_obs,
    a_init=0.5,
    seed=0,
    epochs=10000,
    lr=1e-3,
    n_collocation=300,
    m=0.0,
    omega=0.5,
    lam=2.0,
    w_pde=1.0,
    w_bc=10.0,
    w_data=20.0,
    w_norm=10.0,
    w_reg=1e-4,
    verbose=False
):
    """
    Inverse PINN:
    learns ψ(θ) and estimates spin a_hat jointly.
    """

    set_seed(seed)

    model = MLP().to(DEVICE)

    # Convert initial physical a to raw parameter
    a_init_clamped = min(max(a_init, 1e-4), 0.989)
    raw_value = math.log(a_init_clamped / (0.99 - a_init_clamped))
    raw_a = nn.Parameter(torch.tensor([raw_value], device=DEVICE, dtype=torch.float64))

    optimizer = torch.optim.Adam(list(model.parameters()) + [raw_a], lr=lr)

    theta_f = torch.linspace(1e-4, math.pi - 1e-4, n_collocation, device=DEVICE).view(-1, 1)
    theta_0 = torch.tensor([[1e-4]], device=DEVICE)
    theta_pi = torch.tensor([[math.pi - 1e-4]], device=DEVICE)
    theta_mid = torch.tensor([[math.pi / 2]], device=DEVICE)

    theta_data = theta_data.to(DEVICE)
    psi_obs = psi_obs.to(DEVICE)

    history = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        a_hat = bounded_spin(raw_a)

        res = angular_residual(model, theta_f, a_hat, m=m, omega=omega, lam=lam)
        loss_pde = torch.mean(res**2)

        loss_bc = model(theta_0).pow(2).mean() + model(theta_pi).pow(2).mean()

        pred_data = model(theta_data)
        loss_data = torch.mean((pred_data - psi_obs) ** 2)

        loss_norm = (model(theta_mid) - 1.0).pow(2).mean()

        # Small regularization keeps a within stable optimization range
        loss_reg = a_hat.pow(2).mean()

        loss = (
            w_pde * loss_pde
            + w_bc * loss_bc
            + w_data * loss_data
            + w_norm * loss_norm
            + w_reg * loss_reg
        )

        loss.backward()
        optimizer.step()

        if epoch % 250 == 0 or epoch == epochs - 1:
            history.append({
                "epoch": epoch,
                "total_loss": float(loss.detach().cpu()),
                "pde_loss": float(loss_pde.detach().cpu()),
                "bc_loss": float(loss_bc.detach().cpu()),
                "data_loss": float(loss_data.detach().cpu()),
                "a_hat": float(a_hat.detach().cpu())
            })

        if verbose and epoch % 1000 == 0:
            print(
                f"[INVERSE] epoch={epoch}, "
                f"a_hat={float(a_hat.detach().cpu()):.5f}, "
                f"loss={loss.item():.3e}, data={loss_data.item():.3e}"
            )

    final_a = float(bounded_spin(raw_a).detach().cpu())
    return model, final_a, pd.DataFrame(history)


# ============================================================
# Synthetic observation generation
# ============================================================

def make_synthetic_observations(
    reference_model,
    sigma=0.0,
    n_data=80,
    seed=0
):
    set_seed(seed)

    theta = torch.linspace(1e-4, math.pi - 1e-4, n_data, device=DEVICE).view(-1, 1)

    with torch.no_grad():
        psi_clean = reference_model(theta)

    noise = sigma * torch.randn_like(psi_clean)
    psi_obs = psi_clean + noise

    return theta, psi_clean, psi_obs


# ============================================================
# Main experiment loop
# ============================================================

def run_experiments(
    spin_values=(0.2, 0.3, 0.5, 0.7, 0.9),
    noise_levels=(0.0, 0.01, 0.03, 0.05, 0.10),
    seeds=(0, 1, 2, 3, 4),
    forward_epochs=6000,
    inverse_epochs=8000
):
    all_rows = []
    solution_examples = {}

    for a_true in spin_values:
        print(f"\nGenerating reference solution for a_true={a_true:.2f}")

        ref_model = train_forward_reference(
            a_true=a_true,
            seed=123,
            epochs=forward_epochs,
            verbose=False
        )

        for sigma in noise_levels:
            for seed in seeds:
                print(f"Inverse run: a_true={a_true:.2f}, sigma={sigma:.2f}, seed={seed}")

                theta_data, psi_clean, psi_obs = make_synthetic_observations(
                    ref_model,
                    sigma=sigma,
                    n_data=80,
                    seed=seed
                )

                inv_model, a_hat, hist = train_inverse_pinn(
                    theta_data,
                    psi_obs,
                    a_init=0.5,
                    seed=seed,
                    epochs=inverse_epochs,
                    verbose=False
                )

                abs_error = abs(a_hat - a_true)
                rel_error = abs_error / abs(a_true)

                all_rows.append({
                    "a_true": a_true,
                    "sigma": sigma,
                    "seed": seed,
                    "a_init": 0.5,
                    "a_hat": a_hat,
                    "absolute_error": abs_error,
                    "relative_error": rel_error
                })

                hist.to_csv(
                    os.path.join(
                        OUTDIR,
                        f"history_a{a_true:.2f}_sigma{sigma:.2f}_seed{seed}.csv"
                    ),
                    index=False
                )

                # Save one example per spin at sigma=0.05
                if sigma == 0.05 and seed == seeds[0]:
                    theta_plot = torch.linspace(1e-4, math.pi - 1e-4, 300, device=DEVICE).view(-1, 1)
                    with torch.no_grad():
                        clean_plot = ref_model(theta_plot).detach().cpu().numpy()
                        pred_plot = inv_model(theta_plot).detach().cpu().numpy()

                    solution_examples[a_true] = {
                        "theta": theta_plot.detach().cpu().numpy(),
                        "clean": clean_plot,
                        "pred": pred_plot,
                        "theta_data": theta_data.detach().cpu().numpy(),
                        "psi_obs": psi_obs.detach().cpu().numpy(),
                        "a_hat": a_hat
                    }

    results = pd.DataFrame(all_rows)
    results.to_csv(os.path.join(OUTDIR, "spin_recovery_all_runs.csv"), index=False)

    return results, solution_examples


# ============================================================
# Tables and figures
# ============================================================

def make_tables_and_figures(results, solution_examples):
    # Table: clean spin recovery
    clean = results[results["sigma"] == 0.0].copy()
    clean_summary = clean.groupby("a_true").agg(
        mean_a_hat=("a_hat", "mean"),
        std_a_hat=("a_hat", "std"),
        mae=("absolute_error", "mean"),
        rmse=("absolute_error", lambda x: np.sqrt(np.mean(np.square(x))))
    ).reset_index()

    clean_summary.to_csv(os.path.join(OUTDIR, "table_clean_spin_recovery.csv"), index=False)

    # Table: noise robustness
    noise_summary = results.groupby("sigma").agg(
        mean_a_hat=("a_hat", "mean"),
        std_a_hat=("a_hat", "std"),
        mae=("absolute_error", "mean"),
        rmse=("absolute_error", lambda x: np.sqrt(np.mean(np.square(x))))
    ).reset_index()

    noise_summary.to_csv(os.path.join(OUTDIR, "table_noise_robustness.csv"), index=False)

    # Figure: true spin vs estimated spin
    plt.figure(figsize=(6, 5))
    plt.scatter(results["a_true"], results["a_hat"], alpha=0.6)
    x = np.linspace(0, 1, 100)
    plt.plot(x, x, linestyle="--")
    plt.xlabel("True spin $a_{true}$")
    plt.ylabel("Estimated spin $\\hat{a}$")
    plt.title("Spin recovery: true versus estimated spin")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig_true_vs_estimated_spin.pdf"))
    plt.savefig(os.path.join(OUTDIR, "fig_true_vs_estimated_spin.png"), dpi=300)
    plt.close()

    # Figure: error versus noise
    plt.figure(figsize=(6, 5))
    plt.errorbar(
        noise_summary["sigma"],
        noise_summary["mae"],
        yerr=noise_summary["std_a_hat"],
        marker="o",
        capsize=4
    )
    plt.xlabel("Gaussian noise level $\\sigma$")
    plt.ylabel("Mean absolute spin error")
    plt.title("Spin-estimation error under Gaussian noise")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig_noise_vs_spin_error.pdf"))
    plt.savefig(os.path.join(OUTDIR, "fig_noise_vs_spin_error.png"), dpi=300)
    plt.close()

    # Figure: solution profiles
    for a_true, item in solution_examples.items():
        plt.figure(figsize=(6, 5))
        plt.plot(item["theta"], item["clean"], label="Clean reference")
        plt.scatter(item["theta_data"], item["psi_obs"], s=12, alpha=0.7, label="Noisy samples")
        plt.plot(item["theta"], item["pred"], linestyle="--", label=f"Inverse PINN, $\\hat a$={item['a_hat']:.3f}")
        plt.xlabel("$\\theta$")
        plt.ylabel("$\\psi(\\theta)$")
        plt.title(f"Angular-mode reconstruction for $a_{{true}}$={a_true:.2f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"fig_solution_profile_a{a_true:.2f}.pdf"))
        plt.savefig(os.path.join(OUTDIR, f"fig_solution_profile_a{a_true:.2f}.png"), dpi=300)
        plt.close()

    print("\nSaved tables and figures in:", OUTDIR)
    print("\nClean spin recovery summary:")
    print(clean_summary)
    print("\nNoise robustness summary:")
    print(noise_summary)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    results, solution_examples = run_experiments(
        spin_values=(0.2, 0.3, 0.5, 0.7, 0.9),
        noise_levels=(0.0, 0.01, 0.03, 0.05, 0.10),
        seeds=(0, 1, 2, 3, 4),
        forward_epochs=6000,
        inverse_epochs=8000
    )

    make_tables_and_figures(results, solution_examples)