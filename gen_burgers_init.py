#!/usr/bin/env python
# gen_burgers_init.py
#
# Generate 1D viscous Burgers truth data:
#   u_t + (u^2/2)_x = u_xx   (nu = 1)
#
# Two methods:
#   - imex    : explicit convection + implicit diffusion (Backward Euler)
#   - hopfcole: Hopf–Cole transform with FFT convolution
#
# Time handling:
#   --times auto (default):
#       Use uniform time grid on [0, tfinal] with step auto_dt
#   --times "t0,t1,...":
#       Explicit comma-separated list of times inside [0, tfinal]
#
# Output: compressed .npz saved to ./artifacts/<save_name>
#   x      : [Nx] float32 spatial grid
#   times  : [Nt] float64 time snapshots
#   U      : [Nt, Nx] float32 solution snapshots
#   meta   : dict with PDE / method description
#
# Example:
#   python gen_burgers_init.py --xmax 20 --nx 4096 --tfinal 20 ^
#       --times auto --auto_dt 0.1 --method hopfcole --save truth_burgers_1d.npz

import argparse
import os
import numpy as np


# -------------------------- utilities --------------------------

def parse_times_manual(times_str, tfinal):
    """Parse a comma-separated time list and keep only values in [0,tfinal]."""
    ts = sorted(set(float(s) for s in times_str.split(',')))
    ts = [t for t in ts if 0.0 <= t <= tfinal]
    if len(ts) < 1:
        raise ValueError("Provide at least one snapshot time within [0, tfinal].")
    return np.array(ts, dtype=np.float64)


def build_auto_times(tfinal, auto_dt):
    """
    Build an automatic uniform time grid on [0, tfinal] with step auto_dt.
    """
    tfinal = float(tfinal)
    if tfinal <= 0.0:
        raise ValueError("tfinal must be > 0.")
    if auto_dt <= 0.0:
        raise ValueError("auto_dt must be > 0.")

    n = int(np.floor(tfinal / auto_dt)) + 1
    times = np.linspace(0.0, tfinal, n, dtype=np.float64)

    # Ensure 0 and tfinal are present (guard against float rounding)
    if abs(times[0]) > 1e-12:
        times[0] = 0.0
    if abs(times[-1] - tfinal) > 1e-12:
        times[-1] = tfinal

    return times


def box_initial(x, a=-1.0, b=1.0, amp=1.0):
    """
    Bipolar initial data:
        u0(x) = +amp on (a, 0),
                -amp on (0, b),
                 0   otherwise.
    Default a=-1, b=1, amp=1 gives:
        u0 = +1 on (-1,0), -1 on (0,1).
    """
    u0 = np.zeros_like(x, dtype=np.float64)
    mid = 0.0
    u0[(x > a) & (x < mid)] = float(amp)
    u0[(x > mid) & (x < b)] = -float(amp)
    return u0



# -------------------------- IMEX solver ------------------------

def build_laplacian_tridiag(n, dx, bc):
    """Build 1D Laplacian tridiagonal coefficients with given BC."""
    main = -2.0 * np.ones(n) / dx**2
    off  =  1.0 * np.ones(n - 1) / dx**2
    if bc == 'neumann':
        main[0]  = -1.0 / dx**2
        main[-1] = -1.0 / dx**2
    return main, off


def thomas(main, off, rhs):
    """Solve tridiagonal system with main diagonal 'main' and off-diagonal 'off'."""
    n = len(main)
    a = off.copy()
    b = main.copy()
    c = off.copy()
    d = rhs.astype(np.float64).copy()

    for i in range(1, n):
        w = a[i - 1] / b[i - 1]
        b[i] -= w * c[i - 1]
        d[i] -= w * d[i - 1]

    x = np.zeros_like(d)
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    return x


def upwind_divergence(u, dx):
    """
    Approximate (f(u))_x with f(u) = u^2/2 using sign-based upwind flux.

    u: [n] cell-centered values
    """
    ui = u
    up = np.maximum(ui[:-1], 0.0) * ui[:-1]
    dn = np.minimum(ui[1:],  0.0) * ui[1:]
    F_ip = 0.5 * (up + dn)

    up2 = np.maximum(ui[1:],  0.0) * ui[1:]
    dn2 = np.minimum(ui[:-1], 0.0) * ui[:-1]
    F_im = 0.5 * (dn2 + up2)

    div = np.zeros_like(ui)
    div[1:-1] = (F_ip[1:] - F_im[:-1]) / dx
    div[0]    = (F_ip[0]  - F_im[0])   / dx
    div[-1]   = (F_ip[-1] - F_im[-1])  / dx
    return div


def truth_imex(x, dt, tfinal, capture_times, u0, bc='neumann'):
    """
    IMEX solver: explicit convection, implicit diffusion (Backward Euler).
    bc: 'neumann' or 'dirichlet'.
    """
    n = len(x)
    dx = x[1] - x[0]
    u = u0.astype(np.float64).copy()

    mainL, offL = build_laplacian_tridiag(n, dx, bc)
    # (I - dt*L)
    mainM = 1.0 - dt * mainL
    offM  = -dt * offL

    cap_idx = 0
    captured = []
    times_out = []

    t = 0.0
    capture_times = np.array(capture_times, dtype=np.float64)

    # t=0 snapshot if requested
    if cap_idx < len(capture_times) and abs(capture_times[cap_idx]) < 1e-12:
        captured.append(u.astype(np.float32).copy())
        times_out.append(capture_times[cap_idx])
        cap_idx += 1

    nsteps = int(np.ceil(tfinal / dt))
    for _ in range(nsteps):
        div = upwind_divergence(u, dx)
        rhs = u - dt * div

        if bc == 'dirichlet':
            rhs[0]  = 0.0
            rhs[-1] = 0.0
            mainM0, offM0 = mainM.copy(), offM.copy()
            mainM0[0]  = 1.0
            mainM0[-1] = 1.0
            if len(offM0) > 0:
                offM0[0]  = 0.0
                offM0[-1] = 0.0
            u = thomas(mainM0, offM0, rhs)
        else:
            u = thomas(mainM, offM, rhs)

        if bc == 'dirichlet':
            u[0]  = 0.0
            u[-1] = 0.0

        t += dt
        # record any capture_times that have been crossed
        while cap_idx < len(capture_times) and t + 1e-12 >= capture_times[cap_idx]:
            captured.append(u.astype(np.float32).copy())
            times_out.append(capture_times[cap_idx])
            cap_idx += 1
        if t >= tfinal - 1e-12:
            break

    U = np.stack(captured, axis=0).astype(np.float32)
    T = np.array(times_out, dtype=np.float64)
    if len(T) != U.shape[0]:
        raise RuntimeError(f"IMEX internal mismatch: len(T)={len(T)} vs U.shape[0]={U.shape[0]}")
    return T, U


# -------------------------- Hopf–Cole --------------------------

def _fft_conv_same(a, b):
    """Real 1D convolution (same length) using zero-padding FFT."""
    n = len(a)
    m = len(b)
    L = 1
    while L < n + m - 1:
        L <<= 1
    fa = np.fft.rfft(a, L)
    fb = np.fft.rfft(b, L)
    out = np.fft.irfft(fa * fb, L)
    start = (m - 1) // 2
    return out[start:start + n]


def truth_hopfcole(x, capture_times, u0):
    """
    Hopf–Cole transform for viscous Burgers with nu=1:

        u = -2 (G'_t * w0) / (G_t * w0),
        w0 = exp(-1/2 ∫_{-∞}^x u0).

    On finite domain, approximate ∫_{-∞}^x by cumulative integral from left boundary.
    Convolutions done by FFT with zero-padding to reduce wrap-around error.
    """
    dx = x[1] - x[0]

    # cumulative integral of u0 from left boundary
    U0 = np.cumsum(np.r_[0.0, 0.5 * (u0[:-1] + u0[1:]) * dx])
    w0 = np.exp(-0.5 * U0)
    w0 = np.maximum(w0, 1e-12)

    captured = []
    times_out = []

    capture_times = np.array(capture_times, dtype=np.float64)

    for t in capture_times:
        if abs(t) < 1e-14:
            captured.append(u0.astype(np.float32).copy())
            times_out.append(0.0)
            continue
        # Gaussian kernel and its derivative
        G = (1.0 / np.sqrt(4.0 * np.pi * t)) * np.exp(-(x**2) / (4.0 * t))
        Gp = -(x / (2.0 * t)) * G

        denom = _fft_conv_same(G,  w0)
        numer = _fft_conv_same(Gp, w0)
        u = -2.0 * numer / np.maximum(denom, 1e-12)

        captured.append(u.astype(np.float32))
        times_out.append(t)

    U = np.stack(captured, axis=0).astype(np.float32)
    T = np.array(times_out, dtype=np.float64)
    if len(T) != U.shape[0]:
        raise RuntimeError(f"Hopf–Cole internal mismatch: len(T)={len(T)} vs U.shape[0]={U.shape[0]}")
    return T, U


# -------------------------- main CLI --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xmax',   type=float, default=20.0)
    ap.add_argument('--nx',     type=int,   default=4096)
    ap.add_argument('--dt',     type=float, default=5e-4,
                    help='Time step for IMEX solver (ignored for hopfcole).')
    ap.add_argument('--tfinal', type=float, default=20.0)
    ap.add_argument('--times',  type=str,   default='auto',
                    help="Either 'auto' or comma-separated times in [0,tfinal].")
    ap.add_argument('--auto_dt', type=float, default=0.1,
                    help="Time step for automatic uniform time grid when --times auto.")
    ap.add_argument('--save',   type=str,   default='truth_burgers_1d.npz',
                    help="File name (without path); will be saved under ./artifacts/")
    ap.add_argument('--method', type=str,   default='hopfcole',
                    choices=['imex', 'hopfcole'])
    ap.add_argument('--bc',     type=str,   default='neumann',
                    choices=['neumann', 'dirichlet'],
                    help="Boundary condition for IMEX solver.")
    ap.add_argument('--a',   type=float, default=-1.0,
                    help="Box IC left endpoint.")
    ap.add_argument('--b',   type=float, default= 1.0,
                    help="Box IC right endpoint.")
    ap.add_argument('--amp', type=float, default= 1.0,
                    help="Box IC amplitude.")
    args = ap.parse_args()

    # Spatial grid & initial data
    x = np.linspace(-args.xmax, args.xmax, args.nx, endpoint=True).astype(np.float64)
    u0 = box_initial(x, a=args.a, b=args.b, amp=args.amp)

    # Time snapshots
    if args.times.strip().lower() == 'auto':
        capture_times = build_auto_times(args.tfinal, args.auto_dt)
    else:
        capture_times = parse_times_manual(args.times, args.tfinal)

    # Solve
    if args.method == 'imex':
        T, U = truth_imex(x, args.dt, args.tfinal, capture_times, u0, bc=args.bc)
        meta = dict(
            pde="u_t+(u^2/2)_x=u_xx",
            nu=1.0,
            method="imex",
            scheme="explicit upwind + implicit diffusion (BE)",
            bc=args.bc,
        )
    else:
        T, U = truth_hopfcole(x, capture_times, u0)
        meta = dict(
            pde="u_t+(u^2/2)_x=u_xx",
            nu=1.0,
            method="hopfcole",
            scheme="FFT conv of G_t and G'_t with w0=exp(-1/2∫u0)",
        )

    if len(T) != U.shape[0]:
        raise ValueError(
            f"truth file inconsistent: len(times)={len(T)} but U has {U.shape[0]} frames. Regenerate truth."
        )

    # Ensure artifacts directory exists and build save path
    out_dir = os.path.join(os.getcwd(), "artifacts")
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, args.save)

    np.savez_compressed(
        save_path,
        x=x.astype(np.float32),
        times=T,
        U=U,
        meta=meta,
    )
    print(
        f"Saved {save_path}: U shape={U.shape}, nt={len(T)}, "
        f"first/last t={T[0]:g},{T[-1]:g}, method={args.method}"
    )


if __name__ == '__main__':
    main()
