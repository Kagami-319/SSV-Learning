"""Simple 2D incompressible Navier–Stokes (vorticity form) solver on a periodic box
advanced from t=0 with ETDRK4 (Kassam–Trefethen).

Domain and Variables
--------------------
- Periodic box: [-L, L] × [-L, L] with N×N grid (uniform spacing).
- Vorticity ω(t, x, y), viscosity ν (default 1.0).

Numerics
--------
- Spectral (FFT) for linear terms; 2/3-rule dealiasing.
- ETDRK4 time stepping using precomputed scalar coefficients.
- Biot–Savart in Fourier: ψ̂ = -ω̂ / |k|^2, u = (-ψ_y, ψ_x).
- Nonlinearity computed in real space with finite-difference gradients.

API
---
simulate_vorticity(w0, L=8.0, nu=1.0, dt=5e-3, T=10.0, save_every=10) as default value
    -> (times, x, y, omega)
    - Input: initial vorticity w0 of shape (Ny, Nx) at t=0.
    - Output:
        times: (Nt,) snapshot times from 0 to T,
        x, y: 1D spatial grids,
        omega: (Nt, Ny, Nx) real-valued snapshots.

Caveats
-------
- Choose L large enough to avoid boundary artifacts for the physical window
  R(t) = C * sqrt(1 + t).
"""

import argparse
import os
import numpy as np

def _wavenumbers(N, L):
    """Return 1D wave numbers for domain [-L, L] with N points (period 2L)."""
    k = np.fft.fftfreq(N, d=(2*L)/N) * 2*np.pi
    return k  # shape (N,)

def _dealias_mask(N):
    """2/3 rule mask in 1D."""
    kcut = int(np.floor(N/3))
    m = np.zeros(N, dtype=bool)
    m[:kcut] = True
    m[-kcut:] = True
    return m

def _biot_savart_velocity(w_hat, kx, ky):
    """
    Given vorticity hat (Ny,Nx), compute velocity field u=(u,v) in real space
    via psi_hat = -w_hat / (kx^2+ky^2), u = (-dpsi/dy, dpsi/dx).
    """
    Nx = kx.size; Ny = ky.size
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    K2 = KX**2 + KY**2
    psi_hat = np.zeros_like(w_hat, dtype=np.complex128)
    mask = (K2 != 0.0)
    psi_hat[mask] = -w_hat[mask] / K2[mask]
    # u = (-psi_y, psi_x) = (-i ky psi_hat, i kx psi_hat) in Fourier
    u_hat = -1j * KY * psi_hat
    v_hat =  1j * KX * psi_hat
    u = np.fft.ifft2(u_hat).real
    v = np.fft.ifft2(v_hat).real
    return u, v

def _nonlinear(w, kx, ky, dealias_mask_2d):
    """
    Compute N(w) = - u · grad w in vorticity form (advection of w by u).
    Dealias by 2/3 rule.
    """
    w_hat = np.fft.fft2(w)
    # dealiased state in Fourier before computing u
    w_hat_da = np.zeros_like(w_hat)
    w_hat_da[dealias_mask_2d] = w_hat[dealias_mask_2d]

    u, v = _biot_savart_velocity(w_hat_da, kx, ky)
    wx, wy = np.gradient(w, edge_order=2)
    Nw = -(u*wx + v*wy)
    # return Fourier of nonlinearity with dealias
    Nw_hat = np.fft.fft2(Nw)
    Nw_hat_da = np.zeros_like(Nw_hat)
    Nw_hat_da[dealias_mask_2d] = Nw_hat[dealias_mask_2d]
    return Nw_hat_da

def _etdrk4_coeffs(L_spec, dt, M=32):
    """
    L_spec is spectral linear operator (Ny,Nx) : L = -nu*(kx^2+ky^2)
    Returns arrays E,E2,Q,f1,f2,f3 (Ny,Nx).
    """
    LR = dt * L_spec[..., None]
    j = np.arange(1, M+1)
    r = np.exp(1j*np.pi*(j-0.5)/M)  # M points on unit circle
    LR = LR + r  # broadcast to (..., M)

    E  = np.exp(dt * L_spec)
    E2 = np.exp(0.5*dt * L_spec)

    def mean_over_M(expr):
        return np.mean(expr, axis=-1)

    Q  = dt * mean_over_M((np.exp(LR/2) - 1.0) / LR)
    f1 = dt * mean_over_M((-4 - LR + np.exp(LR)*(4 - 3*LR + LR**2)) / (LR**3))
    f2 = dt * mean_over_M(( 2 + LR + np.exp(LR)*(-2 + LR)) / (LR**3))
    f3 = dt * mean_over_M((-4 - 3*LR - LR**2 + np.exp(LR)*(4 - LR)) / (LR**3))
    return E, E2, Q, f1, f2, f3

def simulate_vorticity(
    w0, L=8.0, nu=1.0, dt=5e-3, T=10.0, save_every=10
):
    """
    Advance w from t=0 to T on periodic box [-L,L]^2 with spacing from w0.
    Returns times (Nt,), xgrid (Nx,), ygrid (Ny,), and omega_snapshots (Nt, Ny, Nx).
    """
    Ny, Nx = w0.shape
    x = np.linspace(-L, L, Nx, endpoint=False)
    y = np.linspace(-L, L, Ny, endpoint=False)
    kx = _wavenumbers(Nx, L)
    ky = _wavenumbers(Ny, L)

    # Dealias mask 2D (2/3 rule)
    mx = _dealias_mask(Nx)
    my = _dealias_mask(Ny)
    dealias_mask_2d = np.outer(my, mx)

    # Spectral linear operator (diffusion): L = -nu*(kx^2+ky^2)
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    K2 = KX**2 + KY**2
    Lspec = -nu * K2

    E, E2, Q, f1, f2, f3 = _etdrk4_coeffs(Lspec, dt, M=32)

    w_hat = np.fft.fft2(w0.copy())

    times = []
    snaps = []

    nsteps = int(np.round(T/dt))
    for n in range(nsteps+1):
        t = n*dt
        if n % save_every == 0:
            times.append(t)
            snaps.append(np.fft.ifft2(w_hat).real.copy())

        # ETDRK4 step in Fourier space
        w = np.fft.ifft2(w_hat).real
        Nv = _nonlinear(w, kx, ky, dealias_mask_2d)
        a_hat = E2*w_hat + Q*Nv
        Na = _nonlinear(np.fft.ifft2(a_hat).real, kx, ky, dealias_mask_2d)
        b_hat = E2*w_hat + Q*Na
        Nb = _nonlinear(np.fft.ifft2(b_hat).real, kx, ky, dealias_mask_2d)
        c_hat = E2*a_hat + Q*(2*Nb - Nv)
        Nc = _nonlinear(np.fft.ifft2(c_hat).real, kx, ky, dealias_mask_2d)

        w_hat = E*w_hat + (f1*Nv + 2*f2*(Na+Nb) + f3*Nc)

    return np.array(times, dtype=np.float64), x, y, np.array(snaps, dtype=np.float32)

"""
Given initial data ω0(x,y), runs the baseline solver, and saves
'artifacts/baseline_omega.npz'.

CLI (examples)
--------------
python gen_ns_init.py --init disk --A1 1 --R1 1 --L 24 --N 768 --T 20 --dt 0.005 --save_every 10 --out artifacts

Arguments
---------
--init: {"disk", "two_blobs"}  (default: "disk")
--A1, --R1: amplitude and radius parameters for the chosen initial condition
--L: half-length of the periodic box (whole domain is [-L, L]^2)
--N: number of grid points per axis
--T, --dt: stopping time and time step (t starts at 0)
--save_every: save every K steps
--out: output directory (in "artifacts")

Outputs
-------
artifacts/baseline_omega.npz containing:
    times (Nt,), x (Nx,), y (Ny,), omega (Nt,Ny,Nx), L (float), N (int)

Notes
-----
- For long-time comparisons with window radius R(t)=C*sqrt(1+t), ensure
  L >= 1.1 * C * sqrt(1 + T_max) to avoid boundary clipping. (Recall that, the ball is actually expanding.)
"""

def make_init_field(kind: str, A1: float, R1: float, L: float, N: int):
    x = np.linspace(-L, L, N, endpoint=False)
    y = np.linspace(-L, L, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='xy')
    if kind == "disk":
        r = np.sqrt(X**2 + Y**2)
        w0 = (r <= R1).astype(np.float32) * A1

    elif kind == "two_blobs":  # two non-radial Gaussians with different peaks/centers
        x1, y1 = -1.5, 0.5
        sigma1 = R1
        A_main = A1
        x2, y2 = 1.0, -0.8
        sigma2 = 1.3 * R1
        A_side = 0.6 * A1

        blob1 = A_main * np.exp(-(((X - x1)**2 + (Y - y1)**2) / sigma1**2))
        blob2 = A_side * np.exp(-(((X - x2)**2 + (Y - y2)**2) / sigma2**2))

        w0 = (blob1 + blob2).astype(np.float32)

    else:
        raise ValueError(f"Unknown init kind: {kind}")
    return w0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", type=str, default="disk")
    ap.add_argument("--A1", type=float, default=1.0)
    ap.add_argument("--R1", type=float, default=1.0)
    ap.add_argument("--L", type=float, default=8.0)
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--T", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=5e-3)
    ap.add_argument("--save_every", type=int, default=10, help="save every K steps")
    ap.add_argument("--out", type=str, default="artifacts")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    w0 = make_init_field(args.init, args.A1, args.R1, args.L, args.N)
    times, x, y, snaps = simulate_vorticity(
        w0, L=args.L, nu=1.0, dt=args.dt, T=args.T, save_every=args.save_every
    )
    path = os.path.join(args.out, "baseline_omega.npz")
    np.savez_compressed(path, times=times, x=x, y=y, omega=snaps, L=args.L, N=args.N)
    print(f"[gen_ns_init] saved baseline to {path} with {len(times)} snapshots (t from 0).")

if __name__ == "__main__":
    main()
