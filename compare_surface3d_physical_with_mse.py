"""
compare_surface3d_physical_interval_mse.py

在 compare_surface3d_physical.py 的基础上：
- 画图仍由 --times 控制
- 新增 --mse_interval a,b：在 [a,b] 上均匀取 N=--mse_interval_n(默认100) 个时刻计算相对RMSE
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse, os, math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- plotting style knobs (paper-friendly) ----
SURF_COLORS = ["#4EA3FF"] * 3  # one unified light blue
SURF_ALPHAS = [0.7, 0.7, 0.7]
EDGE_COLOR  = (0.12, 0.22, 0.35)  # light gray with alpha
EDGE_LW     = 0.18
RS, CS      = 5, 5  # rstride / cstride
VIEW_ELEV, VIEW_AZIM = 25, -60

from nets import make_model, device_select


@torch.no_grad()
def sample_truth_xy(W, times_t, xgrid, ygrid, X, Y, tval):
    """从 baseline snapshot (t,x,y) 双线性插值到 (X,Y) 网格上，并在时间上做线性插值。"""
    import torch.nn.functional as F
    device = X.device
    xmin, xmax = xgrid[0], xgrid[-1]
    ymin, ymax = ygrid[0], ygrid[-1]
    xn = 2.0 * (X - xmin) / (xmax - xmin) - 1.0
    yn = 2.0 * (Y - ymin) / (ymax - ymin) - 1.0
    grid = torch.stack((xn, yn), dim=-1).unsqueeze(0)  # (1,Ny',Nx',2)

    t = torch.tensor([tval], device=device)
    i1 = torch.searchsorted(times_t, t).clamp_(0, len(times_t) - 1)[0]
    i0 = max(int(i1.item()) - 1, 0)
    t0 = float(times_t[i0].item())
    t1 = float(times_t[i1].item())
    a = 0.0 if t1 == t0 else (float(tval) - t0) / (t1 - t0)

    img0 = W[i0:i0 + 1]  # (1,1,Ny,Nx)
    img1 = W[i1:i1 + 1]
    v0 = F.grid_sample(img0, grid, mode='bilinear', align_corners=False, padding_mode='zeros').squeeze().cpu().numpy()
    v1 = F.grid_sample(img1, grid, mode='bilinear', align_corners=False, padding_mode='zeros').squeeze().cpu().numpy()
    return (1.0 - a) * v0 + a * v1


def _parse_interval(s: str):
    """支持 'a,b' 或 'a:b'。"""
    s = s.strip()
    if not s:
        return None
    if "," in s:
        a, b = s.split(",", 1)
    elif ":" in s:
        a, b = s.split(":", 1)
    else:
        raise ValueError(f"--mse_interval expects 'a,b' or 'a:b', got: {s}")
    return float(a), float(b)


@torch.no_grad()
def mse_at_time(
    tval: float,
    *,
    args,
    device,
    # baseline
    W, Tm, Xg, Yg,
    # models
    model_p, model_s
):
    """
    在单个时刻 tval 上：
    - 建物理域圆盘 R(t)=C*sqrt(1+t)
    - truth: sample_truth_xy
    - pred: physical head 用 (x,y,t); ssv head 用 (xi,eta,tau) 再除 (1+t)
    - 返回空间相对RMSE（只在圆盘内），相对量按 truth 的均方 (mean(WT^2)) 归一化
    """
    if tval <= -1.0:
        raise ValueError(f"t must satisfy t > -1 (so that 1+t>0). Got t={tval}")

    R = args.C * math.sqrt(tval + 1.0)
    x = torch.linspace(-R, R, args.Nxy, device=device)
    y = torch.linspace(-R, R, args.Nxy, device=device)
    X, Y = torch.meshgrid(x, y, indexing="xy")
    mask = (X**2 + Y**2) <= R**2  # disk mask, torch bool

    WT = sample_truth_xy(W, Tm, Xg, Yg, X, Y, tval)  # numpy (Nxy,Nxy)
    m_np = mask.cpu().numpy()
    WT[~m_np] = np.nan

    # predictions
    Xi = X / math.sqrt(tval + 1.0)
    Yi = Y / math.sqrt(tval + 1.0)
    tau = torch.full((args.Nxy * args.Nxy,), math.log1p(tval), device=device)
    tvec = torch.full((args.Nxy * args.Nxy,), tval, device=device)

    Ys = model_s(Xi.reshape(-1), Yi.reshape(-1), tau).reshape(args.Nxy, args.Nxy) / (tval + 1.0)
    Yp = model_p(X.reshape(-1),  Y.reshape(-1),  tvec).reshape(args.Nxy, args.Nxy)

    Ys = Ys.detach().cpu().numpy()
    Yp = Yp.detach().cpu().numpy()
    Ys[~m_np] = np.nan
    Yp[~m_np] = np.nan

    # relative MSE: sqrt( mean((pred-truth)^2) / mean(truth^2) )
    denom = float(np.nanmean(WT ** 2))

    mse_p = float(np.nanmean((Yp - WT) ** 2))
    mse_s = float(np.nanmean((Ys - WT) ** 2))

    if (not np.isfinite(denom)) or denom <= 0.0:
        relmse_p = float("nan")
        relmse_s = float("nan")
    else:
        relmse_p = float((mse_p / denom))
        relmse_s = float((mse_s / denom))
    return relmse_p, relmse_s



@torch.no_grad()
def mse_over_interval(
    a: float, b: float, n: int,
    *,
    args, device,
    W, Tm, Xg, Yg,
    model_p, model_s
):
    """在 [a,b] 上均匀取 n 个时刻，分别算相对RMSE，然后给出 per-time 和 mean。"""
    if n <= 1:
        raise ValueError("--mse_interval_n must be >= 2")

    # 数据时间范围：避免 searchsorted 产生外推
    t_data_min = float(Tm[0].item())
    t_data_max = float(Tm[-1].item())

    aa, bb = (a, b) if a <= b else (b, a)
    if aa < t_data_min or bb > t_data_max:
        print(f"[mse_interval] WARNING: interval [{aa},{bb}] exceeds baseline time range "
              f"[{t_data_min},{t_data_max}]. Will clamp to data range to avoid extrapolation.")
    aa = max(aa, t_data_min)
    bb = min(bb, t_data_max)
    if aa > bb:
        raise ValueError(f"After clamping to baseline times, interval becomes empty: [{aa},{bb}]")

    ts = np.linspace(aa, bb, n, endpoint=True, dtype=np.float64)
    rows = []
    for tval in ts:
        mp, ms = mse_at_time(
            float(tval),
            args=args, device=device,
            W=W, Tm=Tm, Xg=Xg, Yg=Yg,
            model_p=model_p, model_s=model_s
        )
        rows.append((float(tval), mp, ms))

    mean_p = float(np.mean([r[1] for r in rows]))
    mean_s = float(np.mean([r[2] for r in rows]))
    return rows, mean_p, mean_s


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

    # plot times
    ap.add_argument("--times", type=str, default="12,16,20")
    ap.add_argument("--out", type=str, default="compare_surface_physical.png")

    # existing: per-plot-time RMSE csv
    ap.add_argument("--mse_out", type=str, default="", help="CSV path to save per-plot-time spatial RMSEs")

    # new: interval relative MSE
    ap.add_argument("--mse_interval", type=str, default="",
                    help="Time interval 'a,b' or 'a:b' to compute relative MSE on uniform time samples (independent of --times).")
    ap.add_argument("--mse_interval_n", type=int, default=100, help="Number of uniform time samples in [a,b].")
    ap.add_argument("--mse_interval_out", type=str, default="", help="CSV path for interval relative RelMSEs")

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
    kw_deep   = dict(
        basis_dim=args.basis_dim,
        width_branch=args.width_branch, depth_branch=args.depth_branch,
        width_trunk=args.width_trunk, depth_trunk=args.depth_trunk
    )
    if args.model in {"concat", "concatmlp"}:
        model_p = make_model(args.model, **kw_concat).to(device)
        model_s = make_model(args.model, **kw_concat).to(device)
    else:
        model_p = make_model("fcnet", **kw_deep).to(device)
        model_s = make_model("fcnet", **kw_deep).to(device)

    model_p.load_state_dict(torch.load(os.path.join(args.artifacts, f"ckpt_{args.model}_physical.pt"), map_location=device))
    model_s.load_state_dict(torch.load(os.path.join(args.artifacts, f"ckpt_{args.model}_ssv.pt"), map_location=device))
    model_p.eval()
    model_s.eval()

    # --- after saving mse_path (CSV), also plot a line chart ---
    def plot_mse_csv(csv_path: str, out_png: str):
        import csv
        import numpy as np
        import matplotlib.pyplot as plt

        ts, mp, ms = [], [], []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # skip rows like "mean,..." if you ever append them
                try:
                    t = float(row["t"])
                    p = float(row["relmse_physical"])
                    s = float(row["relmse_ssv"])
                except Exception:
                    continue
                ts.append(t); mp.append(p); ms.append(s)

        ts = np.asarray(ts); mp = np.asarray(mp); ms = np.asarray(ms)

        plt.figure()
        plt.plot(ts, mp, label="RelMse_physical")
        plt.plot(ts, ms, label="RelMse_ssv")
        plt.xlabel("t")
        plt.ylabel("Relative MSE")
        plt.yscale("log")  # 相对RMSE跨好几个数量级时更清楚；不想用就删掉这行
        plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png, dpi=300)
        plt.close()

    # -------- (A) interval relative MSE (independent of plot times) --------
    if args.mse_interval.strip():
        a, b = _parse_interval(args.mse_interval)
        rows, mean_p, mean_s = mse_over_interval(
            a, b, args.mse_interval_n,
            args=args, device=device,
            W=W, Tm=Tm, Xg=Xg, Yg=Yg,
            model_p=model_p, model_s=model_s
        )
        print(f"[RMSE-INTERVAL] [{a:g},{b:g}] with N={args.mse_interval_n}: "
              f"mean_rel_mse_physical={mean_p:.6e}  mean_rel_mse_ssv={mean_s:.6e}")

        out_path = args.mse_interval_out.strip() or (os.path.splitext(args.out)[0] + f"_rmse_interval_{args.mse_interval_n}.csv")
        try:
            os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("t,relmse_physical,relmse_ssv\n")
                for tval, mp, ms in rows:
                    f.write(f"{tval},{mp},{ms}\n")
                f.write(f"mean,{mean_p},{mean_s}\n")
            print(f"[mse_interval] saved interval relative RelMSEs to {out_path}")
        except Exception as e:
            print(f"[mse_interval] WARNING: failed to save interval relative RelMSEs: {e}")

        # ---- plot interval mse (ONLY) ----
        interval_png = os.path.splitext(out_path)[0] + ".png"
        plot_mse_csv(out_path, interval_png)
        print(f"[mse_interval] saved interval relative MSE plot to {interval_png}")


    # -------- (B) plotting + per-plot-time MSE (original behavior) --------
    tlist = [float(x) for x in args.times.split(",") if x.strip()]
    nrows = len(tlist)  # one row per t

    fig = plt.figure(figsize=(12, 4 * max(nrows, 1)))
    mse_rows = []  # list of (t, mse_physical, mse_ssv)

    for r, tval in enumerate(tlist):
        mp, ms = mse_at_time(
            tval,
            args=args, device=device,
            W=W, Tm=Tm, Xg=Xg, Yg=Yg,
            model_p=model_p, model_s=model_s
        )
        mse_rows.append((tval, mp, ms))
        print(f"[RMSE-PLOT] t={tval:g}  physical={mp:.6e}  ssv={ms:.6e}")

        # 为了画图，我们需要把 truth/pred 再拿出来一次（简单起见复用 mse_at_time 的内部逻辑）
        R = args.C * math.sqrt(tval + 1.0)
        x = torch.linspace(-R, R, args.Nxy, device=device)
        y = torch.linspace(-R, R, args.Nxy, device=device)
        X, Y = torch.meshgrid(x, y, indexing="xy")
        mask = (X**2 + Y**2) <= R**2
        WT = sample_truth_xy(W, Tm, Xg, Yg, X, Y, tval)
        m_np = mask.cpu().numpy()
        WT[~m_np] = np.nan

        Xi = X / math.sqrt(tval + 1.0)
        Yi = Y / math.sqrt(tval + 1.0)
        tau = torch.full((args.Nxy * args.Nxy,), math.log1p(tval), device=device)
        tvec = torch.full((args.Nxy * args.Nxy,), tval, device=device)
        Ys = model_s(Xi.reshape(-1), Yi.reshape(-1), tau).reshape(args.Nxy, args.Nxy) / (tval + 1.0)
        Yp = model_p(X.reshape(-1),  Y.reshape(-1),  tvec).reshape(args.Nxy, args.Nxy)
        Ys = Ys.detach().cpu().numpy()
        Yp = Yp.detach().cpu().numpy()
        Ys[~m_np] = np.nan
        Yp[~m_np] = np.nan

        # Row-wise axis limits: for this tval, keep the same (x,y,z) ranges across the 3 subplots,
        # but allow different t rows to have different ranges.
        try:
            zmin_row = float(np.nanmin([np.nanmin(WT), np.nanmin(Yp), np.nanmin(Ys)]))
            zmax_row = float(np.nanmax([np.nanmax(WT), np.nanmax(Yp), np.nanmax(Ys)]))
        except ValueError:
            zmin_row, zmax_row = 0.0, 1.0  # fallback if everything is NaN

        if not (np.isfinite(zmin_row) and np.isfinite(zmax_row)):
            zmin_row, zmax_row = 0.0, 1.0
        if zmax_row <= zmin_row:
            zpad = 1e-6
        else:
            zpad = 0.02 * (zmax_row - zmin_row)
        zlo, zhi = zmin_row - zpad, zmax_row + zpad

        row_axes = []
        for c, Z in enumerate([WT, Yp, Ys]):
            ax = fig.add_subplot(max(nrows, 1), 3, r * 3 + c + 1, projection="3d")
            ax.plot_surface(
                X.cpu().numpy(), Y.cpu().numpy(), Z,
                rstride=RS, cstride=CS,
                linewidth=EDGE_LW,
                edgecolor=EDGE_COLOR,
                color=SURF_COLORS[c],
                alpha=SURF_ALPHAS[c],
                antialiased=True,
            )
            ax.grid(True, linestyle="--", linewidth=0.35, alpha=0.45)
            row_axes.append(ax)
                        # compact titles; put t-label on the left of each row instead of repeating
            if c == 0:
                ax.set_title("Truth ω(x,y)")
            elif c == 1:
                ax.set_title(f"Physical  (RelMSE={mp:.3e})")
            else:
                ax.set_title(f"SSV  (RelMSE={ms:.3e})")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

            # (x,y): fixed within this row (same t), but varies with t across rows
            ax.set_xlim(-R, R)
            ax.set_ylim(-R, R)

            # (z): fixed within this row (same t), but varies with t across rows
            ax.set_zlim(zlo, zhi)

        # Put the time label on the left of each row (avoid repeating in every panel title)
        if row_axes:
            y0 = min(a.get_position().y0 for a in row_axes)
            y1 = max(a.get_position().y1 for a in row_axes)
            ymid = 0.5 * (y0 + y1)
            fig.text(0.01, ymid, f"t={tval:g}", va="center", ha="left", fontsize=12)


    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.05, top=0.95, wspace=0.25, hspace=0.35)

    # save per-plot-time RMSEs
    mse_path = args.mse_out.strip() or (os.path.splitext(args.out)[0] + "_rmse.csv")
    try:
        os.makedirs(os.path.dirname(mse_path), exist_ok=True) if os.path.dirname(mse_path) else None
        with open(mse_path, "w", encoding="utf-8") as f:
            f.write("t,relmse_physical,relmse_ssv\n")
            for tval, mp, ms in mse_rows:
                f.write(f"{tval},{mp},{ms}\n")
        print(f"[compare_surface3d_physical] saved plot-time RMSEs to {mse_path}")
    except Exception as e:
        print(f"[compare_surface3d_physical] WARNING: failed to save plot-time RMSEs: {e}")

    '''
    # --- after saving mse_path (CSV), also plot a line chart ---
    def plot_mse_csv(csv_path: str, out_png: str):
        import csv
        import numpy as np
        import matplotlib.pyplot as plt

        ts, mp, ms = [], [], []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # skip rows like "mean,..." if you ever append them
                try:
                    t = float(row["t"])
                    p = float(row["relmse_physical"])
                    s = float(row["relmse_ssv"])
                except Exception:
                    continue
                ts.append(t); mp.append(p); ms.append(s)

        ts = np.asarray(ts); mp = np.asarray(mp); ms = np.asarray(ms)

        plt.figure()
        plt.plot(ts, mp, label="RelMse_physical")
        plt.plot(ts, ms, label="RelMse_ssv")
        plt.xlabel("t")
        plt.ylabel("Relative MSE")
        plt.yscale("log")  # 相对RMSE跨好几个数量级时更清楚；不想用就删掉这行
        plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png, dpi=300)
        plt.close()


    mse_png = os.path.splitext(mse_path)[0] + ".png"
    plot_mse_csv(mse_path, mse_png)
    print(f"[compare_surface3d_physical] saved RMSE plot to {mse_png}")
    '''

    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"[compare_surface3d_physical] saved figure to {args.out}")


if __name__ == "__main__":
    main()
