# train_burgers_nets_1d.py
# Train 1D viscous Burgers models in PHYS ((t,x)) and SSV ((tau,xi)) coordinates.
# Model can be either:
#   - FCNet  : branch–trunk factorization with a fixed initial-condition branch vector
#   - Concat : plain coordinate MLP baseline (ConcatMLP)
#
# Usage (examples):
#   FCNet:
#     python train_burgers_1d.py --model fcnet \
#         --truth truth_burgers_1d.npz --t_train_max 10 \
#         --epochs 800 --batch 2048 --nsamples 200000 \
#         --phys_ckpt ckpt_fcnet_physical.pt --ssv_ckpt ckpt_fcnet_ssv.pt
#
#   ConcatMLP:
#     python train_burgers_1d.py --model concat \
#         --truth truth_burgers_1d.npz --t_train_max 10 \
#         --epochs 800 --batch 2048 --nsamples 200000 \
#         --phys_ckpt ckpt_concat_physical.pt --ssv_ckpt ckpt_concat_ssv.pt
#

import argparse, numpy as np, torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# --------------------------- Models ---------------------------

class MLP(nn.Module):
    def __init__(self, in_dim, width=128, depth=4, out_dim=128, act=nn.GELU):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(d, width), act()]
            d = width
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ConcatMLP1D(nn.Module):
    """
    Baseline coordinate MLP:
        coords -> u_hat(t,x) or u_hat(tau,xi)
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


# --------------------------- Dataset --------------------------

class FieldDatasetInterp(Dataset):
    """
    Continuous-in-time sampling on [t_min, t_train_max] with linear interpolation
    between nearest frames; random x per sample.

    If ssv=False:
        coords = [T, X]
    If ssv=True:
        coords = [Tau, Xi] with Tau=log(1+T), Xi=X/sqrt(1+T)
    """
    def __init__(self, x, times, U,
                 t_train_max=10.0,
                 ssv=False,
                 nsamples=200000,
                 seed=0):
        rng = np.random.default_rng(seed)
        self.ssv = ssv

        times = np.asarray(times, dtype=np.float64)
        U = U.astype(np.float32)

        # Restrict to frames <= t_train_max
        mask = times <= t_train_max
        Tw = times[mask]
        Uw = U[mask]
        if len(Tw) < 2:
            raise ValueError("Need at least two frames <= t_train_max for interpolation.")

        # Sample nsamples times uniformly over [Tw[0], t_train_max]
        T = rng.uniform(Tw[0], t_train_max, size=nsamples)

        # Locate bracketing indices j, j+1 such that Tw[j] <= T < Tw[j+1]
        j = np.searchsorted(Tw, T, side='right') - 1
        j = np.clip(j, 0, len(Tw) - 2)
        t0 = Tw[j]
        t1 = Tw[j + 1]
        w  = ((T - t0) / (t1 - t0)).astype(np.float32)  # [nsamples]

        # Sample spatial points
        idx_x = rng.integers(0, len(x), size=nsamples)
        X = x[idx_x].astype(np.float32)

        # Linear interpolation in time at same spatial index
        u0 = Uw[j,     idx_x]
        u1 = Uw[j + 1, idx_x]
        Y  = ((1.0 - w) * u0 + w * u1).astype(np.float32)

        # Build coordinates in PHYS or SSV
        if ssv:
            Tau = np.log1p(T).astype(np.float32)
            Xi  = (X / np.sqrt(1.0 + T)).astype(np.float32)
            self.coords = np.stack([Tau, Xi], axis=1)  # [N,2]
        else:
            self.coords = np.stack([T.astype(np.float32), X], axis=1)  # [N,2]

        self.targets = Y[:, None]   # [N,1]
        self.n = len(self.targets)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.coords[i], self.targets[i]


# --------------------------- Utilities ------------------------

def box_initial(x, a=-1.0, b=1.0, amp=1.0):
    """
    Default Burgers initial condition: box of height amp on (a,b), 0 outside.
    """
    u0 = np.zeros_like(x, dtype=np.float32)
    u0[(x > a) & (x < b)] = float(amp)
    return u0


def build_branch_vec(x_enc, times, U, a=-1.0, b=1.0, amp=1.0):
    """
    Build branch input vector on encoder grid x_enc.

    For robustness, we regenerate the parametric box IC matching burgers_truth_1d default,
    instead of interpolating from the truth file. This keeps PHYS/SSV consistent.
    """
    k0 = np.where(np.isclose(times, 0.0, rtol=0, atol=1e-12))[0]
    if len(k0) > 0:
        # In principle could interpolate U[0] to x_enc,
        # but since truth is generated from the same box IC, we just regenerate it.
        u0 = box_initial(x_enc, a=a, b=b, amp=amp)
    else:
        u0 = box_initial(x_enc, a=a, b=b, amp=amp)
    return u0.astype(np.float32)


def train_one(model, loader, device,
              epochs=800, lr=1e-3,
              mode='phys',
              arch='fcnet',
              branch_vec=None,
              log_interval=50):
    """
    Train one model (FCNet or ConcatMLP) with standard MSE loss.

    arch in {'fcnet', 'concat'}.
    For FCNet we pass a fixed branch_vec (initial condition) and
    call model(branch_vec, coords); for Concat we call model(coords).
    """
    model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 10))

    if arch == 'fcnet':
        if branch_vec is None:
            raise ValueError("branch_vec must be provided when arch='fcnet'")
        bvec = torch.from_numpy(branch_vec[None, :]).to(device)  # [1, B_in]
    else:
        bvec = None

    for ep in range(epochs):
        model.train()
        total = 0.0
        nobs = 0

        for coords, y in loader:
            coords = coords.to(device)  # [B,2]
            y = y.to(device)            # [B,1]
            B = coords.shape[0]

            if arch == 'fcnet':
                pred = model(bvec.expand(B, -1), coords)
            else:
                pred = model(coords)

            loss = F.mse_loss(pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item() * B
            nobs  += B

        sched.step()

        if (ep + 1) % log_interval == 0:
            print(f"[{mode.upper()}-{arch}] Epoch {ep+1}/{epochs}  MSE = {total / nobs:.4e}")

    return model


# --------------------------- Main -----------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--truth', type=str, required=True,
                    help='Path to 1D Burgers truth .npz file.')
    ap.add_argument('--model', type=str, default='fcnet',
                    choices=['fcnet', 'fc', 'concat', 'mlp'],
                    help="Network type: 'fcnet' (factorized) or 'concat' (coordinate MLP).")
    ap.add_argument('--t_train_max', type=float, default=10.0,
                    help='Use only times t <= t_train_max for training window.')
    ap.add_argument('--nsamples', type=int, default=200000,
                    help='Number of training samples (time-space points) per dataset.')
    ap.add_argument('--epochs', type=int, default=800)
    ap.add_argument('--batch', type=int, default=2048)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--width', type=int, default=128)
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--latent', type=int, default=128,
                    help='Latent dim for FCNet (ignored for Concat).')
    ap.add_argument('--encoder_nx', type=int, default=2048,
                    help='Number of encoder grid points for branch input (FCNet).')
    ap.add_argument('--a', type=float, default=-1.0,
                    help='Box IC left endpoint for branch_vec (if t=0 not present).')
    ap.add_argument('--b', type=float, default= 1.0,
                    help='Box IC right endpoint for branch_vec (if t=0 not present).')
    ap.add_argument('--amp', type=float, default= 1.0,
                    help='Box IC amplitude for branch_vec.')
    ap.add_argument('--phys_ckpt', type=str, default='ckpt_phys.pt')
    ap.add_argument('--ssv_ckpt',  type=str, default='ckpt_ssv.pt')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    # Normalize arch name
    mname = args.model.lower()
    if mname in {'fcnet', 'fc'}:
        arch = 'fcnet'
    else:
        arch = 'concat'
    print(f"Using model arch = {arch}")

    # RNG seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load truth
    data = np.load(args.truth, allow_pickle=True)
    x     = data['x'].astype(np.float32)      # [Nx]
    times = data['times'].astype(np.float64)  # [Nt]
    U     = data['U'].astype(np.float32)      # [Nt, Nx]

    # ---- restrict to |x| < 10 to avoid Hopf–Cole boundary artifacts ----
    x_full = x
    U_full = U
    mask = (x_full > -10.0) & (x_full < 10.0)
    x = x_full[mask]           # [N_sub]
    U = U_full[:, mask]        # [Nt, N_sub]

    # Encoder grid for branch input (always built; only FCNet uses it)
    x_enc = np.linspace(float(x.min()), float(x.max()),
                        args.encoder_nx, endpoint=True).astype(np.float32)
    branch_vec = build_branch_vec(x_enc, times, U,
                                  a=args.a, b=args.b, amp=args.amp)  # [encoder_nx]

    # Datasets: continuous-in-time sampling with linear interpolation
    ds_phys = FieldDatasetInterp(
        x, times, U,
        t_train_max=args.t_train_max,
        ssv=False,
        nsamples=args.nsamples,
        seed=args.seed,
    )
    ds_ssv  = FieldDatasetInterp(
        x, times, U,
        t_train_max=args.t_train_max,
        ssv=True,
        nsamples=args.nsamples,
        seed=args.seed + 1,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pin = (device.type == 'cuda')

    dl_phys = DataLoader(ds_phys, batch_size=args.batch,
                         shuffle=True, num_workers=0, pin_memory=pin)
    dl_ssv  = DataLoader(ds_ssv,  batch_size=args.batch,
                         shuffle=True, num_workers=0, pin_memory=pin)

    # ----------------- build models -----------------
    if arch == 'fcnet':
        model_phys = FCNet1D(branch_in=len(x_enc), trunk_in=2,
                             width=args.width, depth=args.depth, latent=args.latent)
        model_ssv  = FCNet1D(branch_in=len(x_enc), trunk_in=2,
                             width=args.width, depth=args.depth, latent=args.latent)
    else:
        model_phys = ConcatMLP1D(in_dim=2, width=args.width, depth=args.depth)
        model_ssv  = ConcatMLP1D(in_dim=2, width=args.width, depth=args.depth)

    # ----------------- Train -----------------
    print("Training PHYS ...")
    train_one(model_phys, dl_phys, device,
              epochs=args.epochs, lr=args.lr,
              mode='phys', arch=arch,
              branch_vec=branch_vec)

    print("Training SSV ...")
    train_one(model_ssv, dl_ssv, device,
              epochs=args.epochs, lr=args.lr,
              mode='ssv', arch=arch,
              branch_vec=branch_vec)

    # ----------------- Save checkpoints -----------------
    torch.save({
        'state_dict': model_phys.state_dict(),
        'arch': arch,
        'x_enc': torch.from_numpy(x_enc),
        'meta': {
            'type': 'phys',
            'coords': '(t,x)' if arch == 'concat' else '(t,x) + branch(box IC)',
            'loss': 'MSE',
            'interp_time': True,
        }
    }, args.phys_ckpt)
    print(f"Saved {args.phys_ckpt}")

    torch.save({
        'state_dict': model_ssv.state_dict(),
        'arch': arch,
        'x_enc': torch.from_numpy(x_enc),
        'meta': {
            'type': 'ssv',
            'coords': '(tau,xi)' if arch == 'concat' else '(tau,xi) + branch(box IC)',
            'loss': 'MSE',
            'interp_time': True,
        }
    }, args.ssv_ckpt)
    print(f"Saved {args.ssv_ckpt}")


if __name__ == '__main__':
    main()
