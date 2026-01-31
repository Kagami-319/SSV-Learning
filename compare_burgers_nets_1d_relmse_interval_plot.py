#!/usr/bin/env python
# compare_burgers_nets_1d_relmse_interval.py
#
# Compute relative MSE (RelMSE) for 1D viscous Burgers, using exactly the same
# evaluation points as the plotting script compare_burgers_nets_1d.py:
#   - spatial grid restricted to |x| < x_clip
#   - truth obtained via linear interpolation in time on that clipped grid
#   - PHYS head evaluated on (t, x)
#   - SSV  head evaluated on (tau, xi) with tau=log(1+t), xi=x/sqrt(1+t)
#
# RelMSE(t) := E[(u_hat(t,x)-u(t,x))^2] / E[u(t,x)^2], expectation over the
# clipped evaluation grid (uniform average over the sampled x points).
#
# Examples:
#   # Discrete times (like the compare/plot script)
#   python compare_burgers_nets_1d_relmse_interval.py \
#       --truth artifacts/truth_burgers_1d.npz \
#       --artifacts artifacts \
#       --model fcnet \
#       --times 12,14,16,18 \
#       --x_clip 8.0 \
#       --relmse_out artifacts/relmse_fcnet_t12_14_16_18.csv
#
#   # Interval evaluation (like your "interval" RMSE/MSE scripts)
#   python compare_burgers_nets_1d_relmse_interval.py \
#       --truth artifacts/truth_burgers_1d.npz \
#       --artifacts artifacts \
#       --model fcnet \
#       --relmse_interval 0,20 \
#       --relmse_interval_n 200 \
#       --x_clip 8.0 \
#       --relmse_out artifacts/relmse_fcnet_0_20_n200.csv
#

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import numpy as np
import torch
import torch.nn as nn
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
        coords -> u_hat
    """
    def __init__(self, in_dim=2, width=128, depth=4):
        super().__init__()
        self.net = MLP(in_dim, width=width, depth=depth, out_dim=1)

    def forward(self, coords):
        return self.net(coords)


class FCNet1D(nn.Module):
    """
    Factorized Coordinate Network (FCNet) for 1D Burgers.
    """
    def __init__(self, branch_in, trunk_in=2, width=128, depth=4, latent=128):
        super().__init__()
        self.branch = MLP(branch_in, width=width, depth=depth, out_dim=latent)
        self.trunk  = MLP(trunk_in,  width=width, depth=depth, out_dim=latent)
        self.bias   = nn.Parameter(torch.zeros(1))

    def forward(self, bvec, coords):
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
    alpha = 0.0 if t1 == t0 else float((t_eval - t0) / (t1 - t0))

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


def rel_mse(u_hat, u_true, eps=1e-12):
    """
    Relative MSE: mean((u_hat-u)^2) / mean(u^2).
    """
    u_hat = np.asarray(u_hat, dtype=np.float64)
    u_true = np.asarray(u_true, dtype=np.float64)
    num = np.mean((u_hat - u_true) ** 2)
    den = np.mean(u_true ** 2)
    return float(num / (den + eps))


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
                    help="Comma-separated list of times, e.g. '12,14,16,18'.")
    ap.add_argument("--relmse_interval", type=str, default="",
                    help="Optional interval 'a,b' to evaluate RelMSE on a linspace. "
                         "If provided, overrides --times.")
    ap.add_argument("--relmse_interval_n", type=int, default=100,
                    help="Number of evaluation times in --relmse_interval (default 100).")
    ap.add_argument("--x_clip", type=float, default=8.0,
                    help="Evaluate only |x| < x_clip (same mask as plotting).")
    ap.add_argument("--width", type=int, default=128,
                    help="Hidden width (must match training).")
    ap.add_argument("--depth", type=int, default=4,
                    help="Hidden depth (must match training).")
    ap.add_argument("--latent", type=int, default=128,
                    help="Latent dim for FCNet (ignored for ConcatMLP).")
    ap.add_argument("--device", type=str, default="cuda",
                    help="'cuda' or 'cpu'.")
    ap.add_argument("--relmse_out", type=str, default="",
                    help="Output CSV path. If empty, defaults to '<model>_relmse.csv' in CWD.")
    ap.add_argument("--relmse_plot_out", type=str, default="",
                    help="Output PNG path for RelMSE(t) scatter+line plot. If empty, defaults to relmse_out with .png extension.")
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

    # restrict to |x| < x_clip (same as plotting script)
    mask = (x_full > -args.x_clip) & (x_full < args.x_clip)
    if not np.any(mask):
        raise ValueError(f"No points with |x| < x_clip={args.x_clip:g}.")
    x_eval = x_full[mask].astype(np.float32)  # evaluation x points

    # ---------------- build / load models ----------------
    mname = args.model.lower()
    if mname in {"fcnet", "fc"}:
        arch = "fcnet"
        base = "fcnet"
    else:
        arch = "concat"
        base = "concat"

    # checkpoint names with _burgers suffix (matches compare_burgers_nets_1d.py)
    phys_ckpt_path = os.path.join(args.artifacts, f"ckpt_{base}_physical_burgers.pt")
    ssv_ckpt_path  = os.path.join(args.artifacts, f"ckpt_{base}_ssv_burgers.pt")
    if not os.path.isfile(phys_ckpt_path):
        raise FileNotFoundError(f"Physical checkpoint not found: {phys_ckpt_path}")
    if not os.path.isfile(ssv_ckpt_path):
        raise FileNotFoundError(f"SSV checkpoint not found: {ssv_ckpt_path}")

    ckpt_p = torch.load(phys_ckpt_path, map_location=device)
    ckpt_s = torch.load(ssv_ckpt_path,  map_location=device)

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
    model_phys.load_state_dict(ckpt_p["state_dict"] if "state_dict" in ckpt_p else ckpt_p)
    model_ssv.load_state_dict(ckpt_s["state_dict"] if "state_dict" in ckpt_s else ckpt_s)
    model_phys.eval()
    model_ssv.eval()

    # ---------------- evaluation times ----------------
    if args.relmse_interval.strip():
        a_str, b_str = [s.strip() for s in args.relmse_interval.split(",")]
        a = float(a_str); b = float(b_str)
        if args.relmse_interval_n < 2:
            raise ValueError("--relmse_interval_n must be >= 2 for an interval.")
        tlist = np.linspace(a, b, int(args.relmse_interval_n), endpoint=True).tolist()
    else:
        tlist = [float(x.strip()) for x in args.times.split(",") if x.strip()]
        if len(tlist) == 0:
            raise ValueError("No evaluation times provided (empty --times and no --relmse_interval).")

    # Torch tensors for evaluation x-grid
    x_eval_t = torch.from_numpy(x_eval).to(device)

    rows = []
    with torch.no_grad():
        for tval in tlist:
            # truth on eval points
            u_true = load_truth_slice(times, U_full, float(tval), mask)  # numpy [Nx_eval]

            # PHYS coords: (t, x)
            T_phys = torch.full_like(x_eval_t, float(tval))
            coords_phys = torch.stack([T_phys, x_eval_t], dim=1).float()

            # SSV coords: (tau, xi)
            tau_val = float(np.log1p(float(tval)))
            xi = (x_eval / np.sqrt(1.0 + float(tval))).astype(np.float32)
            coords_ssv = torch.stack([
                torch.full_like(x_eval_t, tau_val),
                torch.from_numpy(xi).to(device)
            ], dim=1).float()

            # predictions
            if arch == "fcnet":
                B = coords_phys.shape[0]
                u_phys = model_phys(bvec.expand(B, -1), coords_phys).cpu().numpy().squeeze()
                B = coords_ssv.shape[0]
                u_ssv  = model_ssv(bvec.expand(B, -1), coords_ssv).cpu().numpy().squeeze()
            else:
                u_phys = model_phys(coords_phys).cpu().numpy().squeeze()
                u_ssv  = model_ssv(coords_ssv).cpu().numpy().squeeze()

            rp = rel_mse(u_phys, u_true)
            rs = rel_mse(u_ssv,  u_true)
            rows.append((float(tval), rp, rs))

    # ---------------- write CSV ----------------
    out_csv = args.relmse_out.strip()
    if not out_csv:
        tag = f"{base}_relmse.csv"
        out_csv = tag

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("t,relmse_phys,relmse_ssv\n")
        for tval, rp, rs in rows:
            f.write(f"{tval:.10g},{rp:.10g},{rs:.10g}\n")

    print(f"[relmse] saved RelMSE CSV to {out_csv}")

    # ---------------- plot RelMSE vs t (scatter + line) ----------------
    plot_out = args.relmse_plot_out.strip()
    if not plot_out:
        if out_csv.lower().endswith('.csv'):
            plot_out = out_csv[:-4] + '.png'
        else:
            plot_out = out_csv + '.png'

    tvals = [r[0] for r in rows]
    relmse_p = [r[1] for r in rows]
    relmse_s = [r[2] for r in rows]
    fig, ax = plt.subplots()
    ax.plot(tvals, relmse_p, linestyle='-', label='Physical')
    ax.plot(tvals, relmse_s, linestyle='-', label='SSV')
    ax.set_xlabel('t')
    ax.set_ylabel('Relative MSE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    import os as _os
    _os.makedirs(_os.path.dirname(plot_out) or '.', exist_ok=True)
    fig.savefig(plot_out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[relmse] saved RelMSE plot to {plot_out}")
    if args.relmse_interval.strip():
        print(f"[relmse] evaluated on interval t in [{tlist[0]:g}, {tlist[-1]:g}] with n={len(tlist)}")
    else:
        print(f"[relmse] evaluated on times: {', '.join([str(t) for t in tlist])}")


if __name__ == "__main__":
    main()
