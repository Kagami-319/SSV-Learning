import argparse, os, math, json
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from nets import set_seed, device_select, make_model, ConcatMLP, FCNet

# --------------------------- data helpers -----------------------------------

def prepare_baseline_on_device(W_np, xgrid, ygrid, times, device):
    W = torch.from_numpy(W_np).to(device).unsqueeze(1)  # (Nt,1,Ny,Nx)
    X = torch.tensor(xgrid, device=device)
    Y = torch.tensor(ygrid, device=device)
    T = torch.tensor(times,  device=device)
    return W, X, Y, T

@torch.no_grad()
def sample_truth_gpu(W, times_t, xgrid, ygrid, x, y, t):
    device = t.device
    xmin, xmax = xgrid[0], xgrid[-1]
    ymin, ymax = ygrid[0], ygrid[-1]

    xn = 2.0 * (x - xmin) / (xmax - xmin) - 1.0
    yn = 2.0 * (y - ymin) / (ymax - ymin) - 1.0
    grid = torch.stack((xn, yn), dim=-1).view(-1, 1, 1, 2)  # (M,1,1,2)

    i1 = torch.searchsorted(times_t, t).clamp_(0, len(times_t) - 1)
    i0 = (i1 - 1).clamp_(0, len(times_t) - 1)
    t0 = times_t[i0]; t1 = times_t[i1]
    denom = torch.maximum(t1 - t0, torch.tensor(1e-6, device=device))
    a = (t - t0) / denom

    img0 = W[i0]; img1 = W[i1]
    v0 = F.grid_sample(img0, grid, mode='bilinear', align_corners=False, padding_mode='border').view(-1)
    v1 = F.grid_sample(img1, grid, mode='bilinear', align_corners=False, padding_mode='border').view(-1)
    return (1.0 - a) * v0 + a * v1

def sample_batch(M, C, tmin, tmax, device):
    U = torch.rand(M, device=device)
    tau = (math.log1p(tmax) - math.log1p(tmin)) * U + math.log1p(tmin)
    t = torch.expm1(tau)
    U1 = torch.rand(M, device=device); U2 = torch.rand(M, device=device)
    r = C * torch.sqrt(U1); th = 2 * math.pi * U2
    xi1 = r * torch.cos(th); xi2 = r * torch.sin(th)
    R = torch.sqrt(t + 1.0)
    x = R * xi1; y = R * xi2
    return xi1, xi2, tau, t, x, y

# -------------------------------- training ----------------------------------

def train(args):
    set_seed(args.seed)
    device = device_select(args.device)

    data = np.load(os.path.join(args.artifacts, "baseline_omega.npz"))
    times = data["times"].astype(np.float32)
    xgrid = data["x"].astype(np.float32)
    ygrid = data["y"].astype(np.float32)
    W_np  = data["omega"].astype(np.float32)  # (Nt,Ny,Nx)

    tmin = max(args.tmin, float(times[0]))
    tmax = min(args.tmax if args.tmax > 0 else float(times[-1]), float(times[-1]))

    Wgpu, Xg, Yg, Tm = prepare_baseline_on_device(W_np, xgrid, ygrid, times, device)

    if args.model in {"concat", "concatmlp"}:
        model_s = ConcatMLP(width=args.width, depth=args.depth).to(device)   # Ω(τ, ξ)
        model_p = ConcatMLP(width=args.width, depth=args.depth).to(device)   # ω(t, x, y)
    elif args.model in {"fcnet", "fc"}:
        model_s = FCNet(basis_dim=args.basis_dim,
                           width_branch=args.width_branch, depth_branch=args.depth_branch,
                           width_trunk=args.width_trunk,   depth_trunk=args.depth_trunk).to(device)
        model_p = FCNet(basis_dim=args.basis_dim,
                           width_branch=args.width_branch, depth_branch=args.depth_branch,
                           width_trunk=args.width_trunk,   depth_trunk=args.depth_trunk).to(device)
    else:
        raise ValueError(f"Unknown model family: {args.model}")

    if args.compile:
        model_s = torch.compile(model_s)
        model_p = torch.compile(model_p)

    opt = optim.AdamW(list(model_s.parameters()) + list(model_p.parameters()), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    def micro_splits(M, micro):
        if micro <= 0: return [M]
        chunks, done = [], 0
        while done < M:
            k = min(micro, M - done); chunks.append(k); done += k
        return chunks

    model_s.train(); model_p.train()

    for ep in range(1, args.epochs + 1):
        acc_lp = acc_ls = 0.0
        for _ in range(args.iters):
            opt.zero_grad(set_to_none=True)
            tot_lp = tot_ls = 0.0

            for bs in micro_splits(args.M, args.microbatch):
                xi1, xi2, tau, t, x, y = sample_batch(bs, args.C, tmin, tmax, device)
                with torch.no_grad():
                    wt = sample_truth_gpu(Wgpu, Tm, Xg, Yg, x, y, t)   # ω(t,x,y)
                    Omega_t = (t + 1.0) * wt                           # Ω=(1+t)ω

                with torch.cuda.amp.autocast(enabled=args.amp, dtype=(torch.bfloat16 if args.bf16 else torch.float16)):
                    # SSV head Ω(τ, ξ)
                    ys = model_s(xi1, xi2, tau)
                    ls = torch.mean((ys - Omega_t)**2)

                    # Physical head ω(t, x, y)
                    yp = model_p(x, y, t)
                    lp = torch.mean( (yp - wt)**2)

                    loss = ls + lp #Note that the

                scaler.scale(loss).backward()
                tot_ls += float(ls.detach().cpu()); tot_lp += float(lp.detach().cpu())

            scaler.step(opt); scaler.update()
            acc_ls += tot_ls; acc_lp += tot_lp

        print(f"[epoch {ep:03d}] SSV={acc_ls/args.iters:.4e}  PHYS={acc_lp/args.iters:.4e}")

    os.makedirs(args.artifacts, exist_ok=True)
    torch.save(model_s.state_dict(), os.path.join(args.artifacts, f"ckpt_{args.model}_ssv.pt"))
    torch.save(model_p.state_dict(), os.path.join(args.artifacts, f"ckpt_{args.model}_physical.pt"))

    meta = dict(model=args.model, C=args.C, tmin=tmin, tmax=tmax,
                epochs=args.epochs, iters=args.iters, M=args.M,
                basis_dim=args.basis_dim,
                width_branch=args.width_branch, depth_branch=args.depth_branch,
                width_trunk=args.width_trunk,   depth_trunk=args.depth_trunk,
                width=args.width, depth=args.depth)
    with open(os.path.join(args.artifacts, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

# -------------------------------- CLI ---------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=str, default="artifacts")
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--model", type=str, default="fcnet",
                    choices=["concat", "concatmlp", "fcnet", "fc"])

    # concat knobs
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=5)

    # fcnet knobs
    ap.add_argument("--basis_dim", type=int, default=128)
    ap.add_argument("--width_branch", type=int, default=256)
    ap.add_argument("--depth_branch", type=int, default=4)
    ap.add_argument("--width_trunk", type=int, default=256)
    ap.add_argument("--depth_trunk", type=int, default=4)

    # window & weights
    ap.add_argument("--C", type=float, default=4.0)
    ap.add_argument("--tmin", type=float, default=0.0)
    ap.add_argument("--tmax", type=float, default=0.0)

    # schedule
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--M", type=int, default=131072)
    ap.add_argument("--microbatch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=2e-3)

    # accel
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--compile", action="store_true")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    train(args)

if __name__ == "__main__":
    main()

