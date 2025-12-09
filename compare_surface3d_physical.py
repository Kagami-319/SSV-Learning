"""
compare_surface3d_physical.py （For graph plotting only)
3D surface plots in PHYSICAL coordinates (x, y) for:
  [Truth ω(t,x,y), Physical pred ω̂_p mapped to ω, SSV pred→ω̂_s mapped to ω],
where predictions are evaluated at ξ = x / sqrt(1 + t), τ = log(1 + t).

Procedure
---------
- For each t:
  1) Build a physical grid inside disk |(x,y)| <= R(t) with R(t)=C*sqrt(1+t).
  2) Truth: bilinear grid_sample on baseline snapshots at (x,y).
  3) Predictions:
       - map (x,y) → (ξ,τ),
       - evaluate both heads,
       - divide SSV output by (1+t) to restore ω.
- Uses padding_mode='zeros' for out-of-box queries to mimic R^2 decay.

-------------
python compare_surface3d_physical.py --times 12,16,20 --C 4 --Nxy 201 --device cuda --artifacts artifacts --out compare_surface_physical.png

----
- Choose L large enough so R(t) << L for all plotted times to minimize padding.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse, os, math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nets import make_model, device_select

@torch.no_grad()
def sample_truth_xy(W, times_t, xgrid, ygrid, X, Y, tval):
    import torch.nn.functional as F
    device = X.device
    xmin, xmax = xgrid[0], xgrid[-1]
    ymin, ymax = ygrid[0], ygrid[-1]
    xn = 2.0 * (X - xmin) / (xmax - xmin) - 1.0
    yn = 2.0 * (Y - ymin) / (ymax - ymin) - 1.0
    grid = torch.stack((xn, yn), dim=-1).unsqueeze(0)  # (1,Ny',Nx',2)

    # scalar time indices
    t = torch.tensor([tval], device=device)
    i1 = torch.searchsorted(times_t, t).clamp_(0, len(times_t) - 1)[0]
    i0 = max(int(i1.item()) - 1, 0)
    t0 = float(times_t[i0].item()); t1 = float(times_t[i1].item())
    a = 0.0 if t1 == t0 else (float(tval) - t0) / (t1 - t0)

    img0 = W[i0:i0+1]  # (1,1,Ny,Nx)
    img1 = W[i1:i1+1]
    v0 = F.grid_sample(img0, grid, mode='bilinear', align_corners=False, padding_mode='zeros').squeeze().cpu().numpy()
    v1 = F.grid_sample(img1, grid, mode='bilinear', align_corners=False, padding_mode='zeros').squeeze().cpu().numpy()
    return (1.0 - a) * v0 + a * v1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=str, default="artifacts")
    ap.add_argument("--device", type=str, default="cuda")

    # model family + hyperparams (must match training)
    ap.add_argument("--model", type=str, default="fcnet", choices=["concat", "concatmlp", "fcnet", "fc"])
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--basis_dim", type=int, default=128)
    ap.add_argument("--width_branch", type=int, default=256)
    ap.add_argument("--depth_branch", type=int, default=4)
    ap.add_argument("--width_trunk", type=int, default=256)
    ap.add_argument("--depth_trunk", type=int, default=4)

    ap.add_argument("--C", type=float, default=4.0, help="xi-disk radius; physical disk radius is C*sqrt(1+t)")
    ap.add_argument("--Nxy", type=int, default=201)
    ap.add_argument("--times", type=str, default="12,16,20")
    ap.add_argument("--out", type=str, default="compare_surface_physical.png")
    args = ap.parse_args()

    device = device_select(args.device)

    # load baseline
    data = np.load(os.path.join(args.artifacts, "baseline_omega.npz"))
    times = data["times"].astype(np.float32)
    xgrid = data["x"].astype(np.float32)
    ygrid = data["y"].astype(np.float32)
    W_np  = data["omega"].astype(np.float32)

    W  = torch.from_numpy(W_np).to(device).unsqueeze(1)  # (Nt,1,Ny,Nx)
    Xg = torch.tensor(xgrid, device=device)
    Yg = torch.tensor(ygrid, device=device)
    Tm = torch.tensor(times, device=device)

    # instantiate models consistent with training
    kw_concat = dict(width=args.width, depth=args.depth)
    kw_deep   = dict(basis_dim=args.basis_dim, width_branch=args.width_branch, depth_branch=args.depth_branch,
                     width_trunk=args.width_trunk, depth_trunk=args.depth_trunk)
    if args.model in {"concat", "concatmlp"}:
        model_p = make_model(args.model, **kw_concat).to(device)
        model_s = make_model(args.model, **kw_concat).to(device)
    else:
        model_p = make_model("fcnet", **kw_deep).to(device)
        model_s = make_model("fcnet", **kw_deep).to(device)

    model_p.load_state_dict(torch.load(os.path.join(args.artifacts, f"ckpt_{args.model}_physical.pt"), map_location=device))
    model_s.load_state_dict(torch.load(os.path.join(args.artifacts, f"ckpt_{args.model}_ssv.pt"), map_location=device))
    model_p.eval(); model_s.eval()

    tlist = [float(x) for x in args.times.split(",")]
    nrows = len(tlist)
    fig = plt.figure(figsize=(12, 4 * nrows))

    for r, tval in enumerate(tlist):
        # physical window and grid
        R = args.C * math.sqrt(tval + 1.0)
        x = torch.linspace(-R, R, args.Nxy, device=device)
        y = torch.linspace(-R, R, args.Nxy, device=device)
        X, Y = torch.meshgrid(x, y, indexing="xy")
        mask = (X**2 + Y**2) <= R**2  # disk mask

        # truth on physical grid
        WT = sample_truth_xy(W, Tm, Xg, Yg, X, Y, tval)
        WT[~mask.cpu().numpy()] = np.nan

        # predictions:
        Xi = X / math.sqrt(tval + 1.0)
        Yi = Y / math.sqrt(tval + 1.0)
        tau = torch.full((args.Nxy * args.Nxy,), math.log1p(tval), device=device)
        tvec = torch.full((args.Nxy * args.Nxy,), tval, device=device)

        with torch.no_grad():
            Ys = model_s(Xi.reshape(-1), Yi.reshape(-1), tau).reshape(args.Nxy, args.Nxy) / (tval + 1.0)
            Yp = model_p(X.reshape(-1),  Y.reshape(-1),  tvec).reshape(args.Nxy, args.Nxy)

        Ys = Ys.cpu().numpy(); Yp = Yp.cpu().numpy()
        Ys[~mask.cpu().numpy()] = np.nan
        Yp[~mask.cpu().numpy()] = np.nan

        # plot row: truth(x,y), physical pred ω(x,y), ssv pred→ω(x,y)
        for c, Z in enumerate([WT, Yp, Ys]):
            ax = fig.add_subplot(nrows, 3, r * 3 + c + 1, projection="3d")
            surf = ax.plot_surface(
                X.cpu().numpy(), Y.cpu().numpy(), Z,
                rstride=6, cstride=6,
                linewidth=0.3,
                edgecolor="k",
                color="#66CCFF",
                antialiased=True,
            )
            ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
            title = ["Truth ω(x,y)", "Physical pred ω(x,y)", "SSV pred→ω(x,y)"][c]
            ax.set_title(f"t={tval:g}  {title}")
            ax.set_xlabel("x"); ax.set_ylabel("y")

    fig.subplots_adjust(
        left=0.05, right=0.97,
        bottom=0.05, top=0.95,
        wspace=0.25, hspace=0.35
    )
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"[compare_surface3d_physical] saved figure to {args.out}")

if __name__ == "__main__":
    main()
