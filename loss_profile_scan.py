import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ============================================================
# Fixed-spin loss-profile scan for reduced angular Teukolsky-like PINN
# ============================================================
#
# Purpose:
# For each reference spin a_true, fix the candidate spin a_candidate,
# train only the neural-network weights, and record the final profile loss.
#
# Output:
#   loss_profile_results.csv
#   fig_loss_profile_scan.pdf
#
# Notes:
# - This is a diagnostic experiment.
# - The spin is NOT trainable in this scan.
# - If the problem is identifiable, minima should appear near a_true.
# ============================================================

torch.set_default_dtype(torch.float64)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fixed reduced-model parameters
M_AZIMUTHAL = 0.0
OMEGA = 0.5
LAMBDA = 2.0

# Training settings
N_THETA = 200
EPOCHS_REFERENCE = 4000
EPOCHS_PROFILE = 2500
LR = 1e-3

# Reference spins and candidate spins
TRUE_SPINS = [0.20, 0.30, 0.50, 0.70, 0.90]

# You can make this denser later, e.g. np.linspace(0.05, 0.99, 30)
CANDIDATE_SPINS = np.linspace(0.05, 0.99, 20)

# Loss weights
W_PDE = 1.0
W_BC = 1.0
W_DATA = 1.0
W_NORM = 1.0


class PINN(nn.Module):
    def __init__(self, hidden_dim=50, hidden_layers=4):
        super().__init__()

        layers = []
        layers.append(nn.Linear(1, hidden_dim))
        layers.append(nn.Tanh())

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, theta):
        return self.net(theta)


def make_theta_grid(n=N_THETA):
    eps = 1e-5
    theta = torch.linspace(eps, np.pi - eps, n, device=DEVICE).reshape(-1, 1)
    theta.requires_grad_(True)
    return theta


def reduced_residual(model, theta, spin):
    """
    Residual:
    d/dtheta( sin(theta) dpsi/dtheta )
    + [lambda - m^2/sin^2(theta) - a^2 omega^2 cos^2(theta)] psi = 0
    """
    psi = model(theta)

    dpsi = torch.autograd.grad(
        psi,
        theta,
        grad_outputs=torch.ones_like(psi),
        create_graph=True,
        retain_graph=True
    )[0]

    sin_theta = torch.sin(theta)
    flux = sin_theta * dpsi

    dflux = torch.autograd.grad(
        flux,
        theta,
        grad_outputs=torch.ones_like(flux),
        create_graph=True,
        retain_graph=True
    )[0]

    potential = (
        LAMBDA
        - (M_AZIMUTHAL ** 2) / (torch.sin(theta) ** 2)
        - (spin ** 2) * (OMEGA ** 2) * (torch.cos(theta) ** 2)
    )

    residual = dflux + potential * psi
    return residual


def boundary_loss(model):
    theta0 = torch.tensor([[0.0]], device=DEVICE, dtype=torch.float64)
    thetap = torch.tensor([[np.pi]], device=DEVICE, dtype=torch.float64)

    psi0 = model(theta0)
    psip = model(thetap)

    return psi0.pow(2).mean() + psip.pow(2).mean()


def normalization_loss(model):
    theta_mid = torch.tensor([[np.pi / 2]], device=DEVICE, dtype=torch.float64)
    psi_mid = model(theta_mid)
    return (psi_mid - 1.0).pow(2).mean()


def train_reference_solution(a_true):
    """
    Trains a forward PINN with fixed true spin to generate a clean reference profile.
    This is useful if you do not already have saved psi_ref data.
    """
    model = PINN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    theta = make_theta_grid()

    for epoch in range(EPOCHS_REFERENCE):
        optimizer.zero_grad()

        residual = reduced_residual(model, theta, a_true)
        loss_pde = residual.pow(2).mean()
        loss_bc = boundary_loss(model)
        loss_norm = normalization_loss(model)

        loss = W_PDE * loss_pde + W_BC * loss_bc + W_NORM * loss_norm
        loss.backward()
        optimizer.step()

    theta_eval = make_theta_grid()
    with torch.no_grad():
        psi_ref = model(theta_eval).detach().cpu().numpy().reshape(-1)

    theta_np = theta_eval.detach().cpu().numpy().reshape(-1)
    return theta_np, psi_ref


def train_profile_for_candidate(theta_np, psi_ref_np, a_candidate):
    """
    For fixed candidate spin, train only network weights.
    The spin is fixed, not trainable.
    """
    model = PINN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    theta = torch.tensor(theta_np.reshape(-1, 1), device=DEVICE, dtype=torch.float64)
    theta.requires_grad_(True)

    psi_ref = torch.tensor(psi_ref_np.reshape(-1, 1), device=DEVICE, dtype=torch.float64)

    final = {}

    for epoch in range(EPOCHS_PROFILE):
        optimizer.zero_grad()

        psi_pred = model(theta)

        residual = reduced_residual(model, theta, float(a_candidate))
        loss_pde = residual.pow(2).mean()
        loss_data = (psi_pred - psi_ref).pow(2).mean()
        loss_bc = boundary_loss(model)
        loss_norm = normalization_loss(model)

        total_loss = (
            W_PDE * loss_pde
            + W_DATA * loss_data
            + W_BC * loss_bc
            + W_NORM * loss_norm
        )

        total_loss.backward()
        optimizer.step()

        if epoch == EPOCHS_PROFILE - 1:
            final = {
                "total_loss": float(total_loss.detach().cpu()),
                "pde_loss": float(loss_pde.detach().cpu()),
                "data_loss": float(loss_data.detach().cpu()),
                "bc_loss": float(loss_bc.detach().cpu()),
                "norm_loss": float(loss_norm.detach().cpu()),
            }

    return final


def main():
    rows = []

    for a_true in TRUE_SPINS:
        print(f"\nGenerating clean reference profile for a_true={a_true:.2f}")
        theta_np, psi_ref_np = train_reference_solution(a_true)

        for a_candidate in CANDIDATE_SPINS:
            print(f"  profile scan: a_true={a_true:.2f}, a_candidate={a_candidate:.3f}")

            losses = train_profile_for_candidate(theta_np, psi_ref_np, a_candidate)

            rows.append({
                "a_true": a_true,
                "a_candidate": float(a_candidate),
                **losses
            })

    df = pd.DataFrame(rows)
    df.to_csv("loss_profile_results.csv", index=False)

    # Plot total profile loss
    plt.figure(figsize=(7, 5))

    for a_true in TRUE_SPINS:
        sub = df[df["a_true"] == a_true].copy()
        sub = sub.sort_values("a_candidate")

        plt.plot(
            sub["a_candidate"],
            sub["total_loss"],
            marker="o",
            label=f"$a_{{true}}={a_true:.2f}$"
        )

        # Mark true spin
        plt.axvline(a_true, linestyle="--", linewidth=0.8)

    plt.yscale("log")
    plt.xlabel("Fixed candidate spin $a_j$")
    plt.ylabel("Final profile loss")
    plt.title("Fixed-spin loss-profile diagnostic")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_loss_profile_scan.pdf", bbox_inches="tight")
    plt.savefig("fig_loss_profile_scan.png", dpi=300, bbox_inches="tight")

    print("\nSaved:")
    print("  loss_profile_results.csv")
    print("  fig_loss_profile_scan.pdf")
    print("  fig_loss_profile_scan.png")


if __name__ == "__main__":
    main()