#!/usr/bin/env python
# compare_burgers_nets_1d.py
#
# Compare PHYS and SSV models (FCNet or ConcatMLP) against 1D viscous Burgers truth
# at selected times, plotting only |x| < x_clip to avoid FFT edge artifacts.
#
# Example:
#   python compare_burgers_nets_1d.py \
#       --truth artifacts/truth_burgers_1d.npz \
#       --artifacts artifacts \
#       --model fcnet \
#       --times 12,14,16,18 \
#       --x_clip 8.0 \
#       --out artifacts/compare_burgers_fcnet_t12_14_16_18.png
#
#   python compare_burgers_nets_1d.py \
#       --truth artifacts/truth_burgers_1d.npz \
#       --artifacts artifacts \
#       --model concat \
#       --times 12,14,16,18 \
#       --x_clip 8.0 \
#       --out artifacts/compare_burgers_concat_t12_14_16_18.png
#

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------- MLP building block (same as training) ---------------------------

class MLP(nn.Module):
    def __init__(self, in_dim, width=128, depth=4, out_dim=128, act=nn.GELU):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth - 1):
            layers.append(nn.Linear(d, width))
            layers.append(act())
            d = width
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# --------------------------- Models (must match training) ---------------------------

class ConcatMLP1D(nn.Module):
    """
    Baseline coordinate MLP:
        coords -> u_hat(t,x) or u_hat(tau,xi)

    This matches the definition used in train_fcnet_burgers_1d.py:
        self.net = MLP(in_dim, width=..., depth=..., out_dim=1)
    so the parameter keys are 'net.net.0.weight', etc.
    """
    def __init__(self, in_dim=2, width=128, depth=4):
        super().__init__()
        self.net = MLP(in_dim, width=width, depth=depth, out_dim=1)

    def forward(self, coords):
        return self.net(coords)


class FCNet1D(nn.Module):
    """
    Factorized Coordinate Network (FCNet) for 1D Burgers.

    Branch encodes initial condition samples (fixed vector).
    Trunk encodes coordinates:
      - PHYS: (t, x)
      - SSV : (tau, xi) with tau=log(1+t), xi=x/sqrt(1+t)
    Output: inner product + learnable bias (scalar).

    This matches the definition used in train_fcnet_burgers_1d.py.
    """
    def __init__(self, branch_in, trunk_in=2, width=128, depth=4, latent=128):
        super().__init__()
        self.branch = MLP(branch_in, width=width, depth=depth, out_dim=latent)
        self.trunk  = MLP(trunk_in,  width=width, depth=depth, out_dim=latent)
        self.bias   = nn.Parameter(torch.zeros(1))

    def forward(self, bvec, coords):
        # bvec: [B, branch_in], coords: [B, trunk_in]
        phi_b = self.branch(bvec)
        phi_t = self.trunk(coords)
        return (phi_b * phi_t).sum(dim=-1, keepdim=True) + self.bias


# --------------------------- Helpers ---------------------------

def load_truth_slice(times, U_full, t_eval, mask):
    """
    Linearly interpolate truth at time t_eval on the clipped x-grid (mask).
    """
    times = np.asarray(times, dtype=np.float64)
    if t_eval < times[0] or t_eval > times[-1]:
        raise ValueError(
            f"Requested time t={t_eval:g} outside truth range "
            f"[{times[0]:g}, {times[-1]:g}]."
        )

    j = np.searchsorted(times, t_eval, side="right") - 1
    j = np.clip(j, 0, len(times) - 2)
    t0 = times[j]
    t1 = times[j + 1]
    if t1 == t0:
        alpha = 0.0
    else:
        alpha = float((t_eval - t0) / (t1 - t0))

    u0 = U_full[j, mask]
    u1 = U_full[j + 1, mask]
    return (1.0 - alpha) * u0 + alpha * u1


def box_initial(x, a=-1.0, b=1.0, amp=1.0):
    """
    Default Burgers initial condition used for FCNet branch_vec in training.
    """
    u0 = np.zeros_like(x, dtype=np.float32)
    u0[(x > a) & (x < b)] = float(amp)
    return u0


def build_branch_vec_from_x_enc(x_enc, a=-1.0, b=1.0, amp=1.0):
    """
    Rebuild the branch input vector exactly as in training: box initial condition
    sampled on encoder grid x_enc.
    """
    return box_initial(x_enc, a=a, b=b, amp=amp).astype(np.float32)


# --------------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", type=str, required=True,
                    help="Path to 1D Burgers truth .npz (with x, times, U).")
    ap.add_argument("--artifacts", type=str, default="artifacts",
                    help="Directory containing checkpoints.")
    ap.add_argument("--model", type=str, default="fcnet",
                    choices=["fcnet", "fc", "concat", "concatmlp", "mlp"],
                    help="Network type used in training (controls checkpoint names).")
    ap.add_argument("--times", type=str, default="12,14,16,18",
                    help="Comma-separated list of times to compare, e.g. '12,14,16,18'.")
    ap.add_argument("--x_clip", type=float, default=8.0,
                    help="Plot only |x| < x_clip to avoid FFT edge artifacts.")
    ap.add_argument("--width", type=int, default=128,
                    help="Hidden width (must match training).")
    ap.add_argument("--depth", type=int, default=4,
                    help="Hidden depth (must match training).")
    ap.add_argument("--latent", type=int, default=128,
                    help="Latent dim for FCNet (ignored for ConcatMLP).")
    ap.add_argument("--device", type=str, default="cuda",
                    help="'cuda' or 'cpu'.")
    ap.add_argument("--out", type=str,
                    default="compare_burgers_1d.png",
                    help="Output PNG path.")
    args = ap.parse_args()

    # device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # ---------------- load truth ----------------
    data = np.load(args.truth, allow_pickle=True)
    x_full = data["x"].astype(np.float32)      # [Nx]
    times  = data["times"].astype(np.float64)  # [Nt]
    U_full = data["U"].astype(np.float32)      # [Nt, Nx]

    # restrict to |x| < x_clip
    mask = (x_full > -args.x_clip) & (x_full < args.x_clip)
    if not np.any(mask):
        raise ValueError(f"No points with |x| < x_clip={args.x_clip:g}.")
    x_plot = x_full[mask]  # [Nx_clip]

    # ---------------- build / load models ----------------
    mname = args.model.lower()
    if mname in {"fcnet", "fc"}:
        arch = "fcnet"
        base = "fcnet"
    else:
        arch = "concat"
        base = "concat"

    # ckpt names带 _burgers 后缀，避免和 NS 冲突
    phys_ckpt_path = os.path.join(args.artifacts, f"ckpt_{base}_physical_burgers.pt")
    ssv_ckpt_path  = os.path.join(args.artifacts, f"ckpt_{base}_ssv_burgers.pt")

    if not os.path.isfile(phys_ckpt_path):
        raise FileNotFoundError(f"Physical checkpoint not found: {phys_ckpt_path}")
    if not os.path.isfile(ssv_ckpt_path):
        raise FileNotFoundError(f"SSV checkpoint not found: {ssv_ckpt_path}")

    ckpt_p = torch.load(phys_ckpt_path, map_location=device)
    ckpt_s = torch.load(ssv_ckpt_path,  map_location=device)

    # Construct models consistent with training
    if arch == "fcnet":
        if "x_enc" not in ckpt_p:
            raise KeyError("FCNet checkpoints must contain 'x_enc' for the branch input grid.")
        x_enc = ckpt_p["x_enc"].cpu().numpy().astype(np.float32)  # [Ne]
        branch_vec = build_branch_vec_from_x_enc(x_enc)           # [Ne]

        model_phys = FCNet1D(branch_in=len(x_enc), trunk_in=2,
                             width=args.width, depth=args.depth,
                             latent=args.latent).to(device)
        model_ssv  = FCNet1D(branch_in=len(x_enc), trunk_in=2,
                             width=args.width, depth=args.depth,
                             latent=args.latent).to(device)

        bvec = torch.from_numpy(branch_vec[None, :]).to(device)  # [1, Ne]
    else:
        model_phys = ConcatMLP1D(in_dim=2, width=args.width, depth=args.depth).to(device)
        model_ssv  = ConcatMLP1D(in_dim=2, width=args.width, depth=args.depth).to(device)
        bvec = None

    # load weights
    if "state_dict" in ckpt_p:
        model_phys.load_state_dict(ckpt_p["state_dict"])
    else:
        model_phys.load_state_dict(ckpt_p)
    if "state_dict" in ckpt_s:
        model_ssv.load_state_dict(ckpt_s["state_dict"])
    else:
        model_ssv.load_state_dict(ckpt_s)

    model_phys.eval()
    model_ssv.eval()

    # ---------------- evaluation times ----------------
    tlist = [float(x.strip()) for x in args.times.split(",") if x.strip()]

    fig, axes = plt.subplots(len(tlist), 1,
                             figsize=(6, 2.5 * len(tlist)),
                             sharex=True)
    if len(tlist) == 1:
        axes = [axes]

    x_plot_torch = torch.from_numpy(x_plot).to(device)

    with torch.no_grad():
        for i, tval in enumerate(tlist):
            ax = axes[i]

            u_truth = load_truth_slice(times, U_full, tval, mask)

            T_phys = torch.full_like(x_plot_torch, float(tval))
            coords_phys = torch.stack([T_phys, x_plot_torch], dim=1)

            tau_val = float(np.log1p(tval))
            Xi = (x_plot / np.sqrt(1.0 + tval)).astype(np.float32)
            coords_ssv = torch.stack([
                torch.full_like(x_plot_torch, tau_val),
                torch.from_numpy(Xi).to(device)
            ], dim=1).float()

            if arch == "fcnet":
                B = coords_phys.shape[0]
                u_phys = model_phys(bvec.expand(B, -1), coords_phys).cpu().numpy().squeeze()
                B = coords_ssv.shape[0]
                u_ssv = model_ssv(bvec.expand(B, -1), coords_ssv).cpu().numpy().squeeze()
            else:
                u_phys = model_phys(coords_phys).cpu().numpy().squeeze()
                u_ssv = model_ssv(coords_ssv).cpu().numpy().squeeze()

            ax.plot(x_plot, u_truth, "k-", label="Truth")
            ax.plot(x_plot, u_phys, "b--", label="PHYS")
            ax.plot(x_plot, u_ssv, "r-.", label="SSV")

            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(loc="upper right", fontsize=8)

            ax.set_xlabel("")
            ax.set_ylabel("")

            ax.text(0.0, 1.02, r"$u$",
                    transform=ax.transAxes,
                    ha="left", va="bottom")

            ax.text(1.0, -0.15, r"$x$",
                    transform=ax.transAxes,
                    ha="right", va="top")

            ax.text(0.5, -0.30, rf"$t = {tval:g}$",
                    transform=ax.transAxes,
                    ha="center", va="top")

    fig.subplots_adjust(hspace=0.6, bottom=0.30)

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare_burgers_nets_1d] saved figure to {out_path}")


if __name__ == "__main__":
    main()
