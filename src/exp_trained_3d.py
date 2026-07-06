"""EXP -- LARGE TRAINED 3D PINN diagnostic on the GPU (Poisson + stiff anisotropic).

Purpose. Two tests at scale:
  (i)  large trained three-dimensional sweep: a large tanh
       PINN (6 hidden layers x 128, 8000 interior + 2000 boundary collocation,
       600 warmup + 1200 Adam iterations, 8 seeds) is trained on BOTH a 3D Poisson
       problem and a 3D stiff-anisotropic elliptic problem, sweeping the boundary
       weight over several decades, WITH a hard-constrained twin (lifting) at
       matched budget.
  (ii) wall-clock: every PINN train and every finite-difference reference solve
       is timed. The finite-difference solve provides (a) the reference field for
       the relative-L2 error metric and (b) timing context.

Problems (reuse of exp_diagnostic_3d definitions).
  P1  3D Poisson : -Lap u = f on the unit cube, u = sin(2.4 pi x)sin(1.7 pi y)
                   sin(1.3 pi z), Dirichlet trace on the six faces.
  P2  3D stiff   : -(u_xx + 40 u_yy + 1600 u_zz) = f, same manufactured u.
The manufactured u has non-integer frequencies, so it is not grid-representable:
the finite-difference reference itself carries a discretization error, which is
reported alongside the PINN error.

Reference truth (conventional solver). 7-point structured finite differences on
an N^3 grid, assembled as a scipy CSR sparse operator and solved by a sparse
direct factorization (scipy.sparse.linalg.spsolve / splu). The exact Dirichlet
trace is imposed on the boundary nodes. The PINN field is interpolated onto the
same grid (trilinear) and the relative-L2 error is measured there.

Transfer test (prospective). At a fixed short warmup point the tangent-kernel
constants (rho, mu, nu, sigma_B, kappa_red) are MEASURED kernel-side, and the
predicted conditioning-valley weight, valley floor, product limit and hard-twin
call are COMMITTED (printed + written to disk) BEFORE the confirming sweep runs.
Then the sweep and hard twin are run and scored against the commitment. The
report scores how the fixed-feature diagnostic transfers to the trained 3D
network at scale.

Deterministic seeds; GPU (device='cuda'); sized to stay under 14 GB of device
memory and about 25 min wall time on a single consumer GPU. Writes data/trained_3d.json.
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = Path(__file__).resolve().parent.parent / "data" / "trained_3d.json"

# --------------------------------------------------------------------------- #
# Fixed, pre-registered protocol constants.                                   #
# --------------------------------------------------------------------------- #
DEV = "cuda"
DTYPE = torch.float32
WIDTH, DEPTH = 128, 6           # 6 hidden layers x 128
N_INT = 8000                    # interior collocation points
N_BND = 2000                    # boundary collocation points (six faces)
WARM_STEPS = 600                # fixed warmup budget for the measurement point
SWEEP_STEPS = 1200              # Adam iters per confirming train (as recorded in data/trained_3d.json)
LAM_WARM = 100.0                # neutral warmup weight
SWEEP_LAMS = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5]   # six decades of boundary weight
SWEEP_SEEDS = 8                 # seeds per soft weight
HARD_SEEDS = 8                  # seeds for the hard twin (to expose basin-escape fragility)
FD_N = 49                       # finite-difference grid: FD_N^3 interior unknowns

# manufactured target frequencies (non-integer -> not grid-representable)
KX, KY, KZ = 2.4, 1.7, 1.3
# problem operator anisotropies
A_POISSON = (1.0, 1.0, 1.0)
A_STIFF = (1.0, 40.0, 1600.0)


# --------------------------------------------------------------------------- #
# Manufactured solution and right-hand sides (torch and numpy).               #
# --------------------------------------------------------------------------- #
def u_exact_t(X):
    return (torch.sin(KX * math.pi * X[:, 0]) * torch.sin(KY * math.pi * X[:, 1])
            * torch.sin(KZ * math.pi * X[:, 2]))


def u_exact_np(X):
    return (np.sin(KX * np.pi * X[:, 0]) * np.sin(KY * np.pi * X[:, 1])
            * np.sin(KZ * np.pi * X[:, 2]))


def rhs_coef(a):
    ax, ay, az = a
    return (ax * KX ** 2 + ay * KY ** 2 + az * KZ ** 2) * math.pi ** 2


def f_rhs_t(X, a):
    # L u = -(ax u_xx + ay u_yy + az u_zz) = rhs_coef * u  for this u
    return rhs_coef(a) * u_exact_t(X)


# --------------------------------------------------------------------------- #
# Network and hard-constrained lifting.                                       #
# --------------------------------------------------------------------------- #
class MLP(torch.nn.Module):
    def __init__(self, width=WIDTH, depth=DEPTH):
        super().__init__()
        layers, d = [], 3
        for _ in range(depth):
            layers += [torch.nn.Linear(d, width), torch.nn.Tanh()]
            d = width
        layers += [torch.nn.Linear(d, 1)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, X):
        return self.net(X)


def transfinite_lifting_3d(X):
    """Transfinite interpolation of the exact Dirichlet data on the cube.

    Standard 3D transfinite interpolation: sum of face terms minus edge terms
    plus vertex terms, so that on each face it reproduces the exact boundary
    trace of u_exact. Because u_exact vanishes on all six faces (the chosen
    manufactured target is zero on the cube boundary), this lifting is exactly
    zero; the bubble N carries the whole solution. It is written in full so the
    construction is correct for a general nonzero trace as well.
    """
    x, y, z = X[:, 0:1], X[:, 1:2], X[:, 2:3]

    def g(xx, yy, zz):
        return (torch.sin(KX * math.pi * xx) * torch.sin(KY * math.pi * yy)
                * torch.sin(KZ * math.pi * zz))

    o = torch.ones_like(x)
    zc = torch.zeros_like(x)
    # face contributions (fix one coordinate to 0 or 1)
    F = ((1 - x) * g(zc, y, z) + x * g(o, y, z)
         + (1 - y) * g(x, zc, z) + y * g(x, o, z)
         + (1 - z) * g(x, y, zc) + z * g(x, y, o))
    # edge contributions (fix two coordinates)
    E = ((1 - x) * (1 - y) * g(zc, zc, z) + (1 - x) * y * g(zc, o, z)
         + x * (1 - y) * g(o, zc, z) + x * y * g(o, o, z)
         + (1 - y) * (1 - z) * g(x, zc, zc) + (1 - y) * z * g(x, zc, o)
         + y * (1 - z) * g(x, o, zc) + y * z * g(x, o, o)
         + (1 - x) * (1 - z) * g(zc, y, zc) + (1 - x) * z * g(zc, y, o)
         + x * (1 - z) * g(o, y, zc) + x * z * g(o, y, o))
    # vertex contributions (fix all three)
    V = ((1 - x) * (1 - y) * (1 - z) * g(zc, zc, zc)
         + x * (1 - y) * (1 - z) * g(o, zc, zc)
         + (1 - x) * y * (1 - z) * g(zc, o, zc)
         + (1 - x) * (1 - y) * z * g(zc, zc, o)
         + x * y * (1 - z) * g(o, o, zc)
         + x * (1 - y) * z * g(o, zc, o)
         + (1 - x) * y * z * g(zc, o, o)
         + x * y * z * g(o, o, o))
    return (F - E + V).squeeze(-1)


def bubble(X):
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    return 64.0 * x * (1 - x) * y * (1 - y) * z * (1 - z)


def predict_field(model, X, hard):
    """Network prediction on points X (no grad), hard or soft parameterization."""
    if hard:
        return transfinite_lifting_3d(X) + bubble(X) * model(X).squeeze(-1)
    return model(X).squeeze(-1)


def anisotropic_laplacian(model, X, a, hard):
    """Return L u = -(ax u_xx + ay u_yy + az u_zz) for the (hard/soft) field."""
    ax, ay, az = a
    X = X.requires_grad_(True)
    if hard:
        u = transfinite_lifting_3d(X) + bubble(X) * model(X).squeeze(-1)
    else:
        u = model(X).squeeze(-1)
    grads = torch.autograd.grad(u.sum(), X, create_graph=True)[0]
    uxx = torch.autograd.grad(grads[:, 0].sum(), X, create_graph=True)[0][:, 0]
    uyy = torch.autograd.grad(grads[:, 1].sum(), X, create_graph=True)[0][:, 1]
    uzz = torch.autograd.grad(grads[:, 2].sum(), X, create_graph=True)[0][:, 2]
    return u, -(ax * uxx + ay * uyy + az * uzz)


# --------------------------------------------------------------------------- #
# Collocation batches.                                                        #
# --------------------------------------------------------------------------- #
def cube_boundary_points(gen, b):
    face = torch.randint(0, 6, (b,), generator=gen)
    Xb = torch.rand((b, 3), generator=gen)
    axis = face // 2
    val = (face % 2).to(Xb.dtype)
    idx = torch.arange(b)
    Xb[idx, axis] = val
    return Xb


def make_batch(seed):
    gen = torch.Generator(device="cpu").manual_seed(seed + 1)
    Xi = torch.rand((N_INT, 3), generator=gen).to(DEV, DTYPE)
    Xb = cube_boundary_points(gen, N_BND).to(DEV, DTYPE)
    return Xi, Xb


# --------------------------------------------------------------------------- #
# Finite-difference reference solver (conventional; reference truth + timing). #
# --------------------------------------------------------------------------- #
def fd_reference(a, n=FD_N):
    """7-point finite-difference solve of L u = f with exact Dirichlet trace.

    Interior grid of n^3 unknowns on a uniform mesh of the unit cube; the six
    faces carry the exact Dirichlet data of u_exact. Assembled as a CSR sparse
    matrix, solved by a sparse LU factorization (scipy splu). Returns the full
    (n+2)^3 solution field on grid coordinates, the wall time, and the reference
    error of the discretization against the true continuous u (context: the
    manufactured u is not grid-representable, so this is nonzero).
    """
    ax, ay, az = a
    h = 1.0 / (n + 1)
    coords = (np.arange(n + 2)) * h                 # 0 .. 1 including boundary
    N = n ** 3
    cx, cy, cz = ax / h ** 2, ay / h ** 2, az / h ** 2
    diag = 2.0 * (cx + cy + cz)

    # interior node coordinates and flat index map (vectorized assembly)
    ii, jj, kk = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    ii, jj, kk = ii.ravel(), jj.ravel(), kk.ravel()
    p = (ii * n + jj) * n + kk
    xi_all = coords[ii + 1]; yj_all = coords[jj + 1]; zk_all = coords[kk + 1]
    Xi = np.stack([xi_all, yj_all, zk_all], axis=1)

    rows = [p]; cols = [p]; vals = [np.full(N, diag)]
    b = np.zeros(N)
    for axis_idx, (idx, cc) in enumerate(((ii, cx), (jj, cy), (kk, cz))):
        for d in (-1, 1):
            nb = idx + d
            interior = (nb >= 0) & (nb < n)
            # off-diagonal couplings to interior neighbors
            if axis_idx == 0:
                q = ((ii + d) * n + jj) * n + kk
            elif axis_idx == 1:
                q = (ii * n + (jj + d)) * n + kk
            else:
                q = (ii * n + jj) * n + (kk + d)
            rows.append(p[interior]); cols.append(q[interior])
            vals.append(np.full(interior.sum(), -cc))
            # boundary neighbors move to the RHS with exact Dirichlet data
            bmask = ~interior
            if bmask.any():
                Xb = Xi[bmask].copy()
                Xb[:, axis_idx] = coords[idx[bmask] + 1 + d]
                np.add.at(b, p[bmask], cc * u_exact_np(Xb))
    rows = np.concatenate(rows); cols = np.concatenate(cols); vals = np.concatenate(vals)
    f = rhs_coef(a) * u_exact_np(Xi)
    b = b + f
    Amat = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    t0 = time.time()
    # sparse DIRECT solve. The 7-point 3D operator is symmetric; the default
    # COLAMD ordering fills in catastrophically in 3D, so the symmetric-structure
    # ordering MMD_AT_PLUS_A is used (about 5x faster factorization at this size).
    lu = spla.splu(sp.csc_matrix(Amat), permc_spec="MMD_AT_PLUS_A")
    u_int = lu.solve(b)
    fd_wall = time.time() - t0
    # reference discretization error vs the continuous exact u (context)
    u_true_int = u_exact_np(Xi)
    fd_ref_err = float(np.linalg.norm(u_int - u_true_int) / np.linalg.norm(u_true_int))
    # embed into full field including boundary faces for trilinear interp
    U = np.empty((n + 2, n + 2, n + 2))
    # fill interior
    U[1:-1, 1:-1, 1:-1] = u_int.reshape(n, n, n)
    # fill boundary faces with exact data
    gx, gy, gz = np.meshgrid(coords, coords, coords, indexing="ij")
    Uexact_grid = (np.sin(KX * np.pi * gx) * np.sin(KY * np.pi * gy)
                   * np.sin(KZ * np.pi * gz))
    U[0, :, :] = Uexact_grid[0, :, :]
    U[-1, :, :] = Uexact_grid[-1, :, :]
    U[:, 0, :] = Uexact_grid[:, 0, :]
    U[:, -1, :] = Uexact_grid[:, -1, :]
    U[:, :, 0] = Uexact_grid[:, :, 0]
    U[:, :, -1] = Uexact_grid[:, :, -1]
    return dict(U=U, coords=coords, n=n, wall_s=fd_wall, ref_err=fd_ref_err, nnz=int(Amat.nnz))


def real_rel_l2(model, fd, hard):
    """Evaluate the PINN on the FD grid and measure relative-L2 vs FD truth.

    The FD solution field is the reference truth; the continuous exact field is
    also available for context. The PINN is queried directly at the FD grid nodes
    on the GPU (no interpolation needed, since the network is a continuous field
    evaluable at any point). The relative-L2 error is measured against the FD
    field (the conventional reference) on the interior nodes where FD solved, and
    also against the continuous exact u for context.
    """
    coords = fd["coords"]
    n2 = coords.size
    gx, gy, gz = np.meshgrid(coords, coords, coords, indexing="ij")
    P = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    with torch.no_grad():
        Xt = torch.tensor(P, device=DEV, dtype=DTYPE)
        # chunk to bound memory
        out = []
        for s in range(0, Xt.shape[0], 200000):
            out.append(predict_field(model, Xt[s:s + 200000], hard).cpu().numpy())
        u_pinn = np.concatenate(out).reshape(n2, n2, n2)
    U = fd["U"]
    # measure on interior nodes (where FD actually solved)
    up = u_pinn[1:-1, 1:-1, 1:-1].ravel()
    uf = U[1:-1, 1:-1, 1:-1].ravel()
    rel_fd = float(np.linalg.norm(up - uf) / np.linalg.norm(uf))
    # also vs continuous exact for context (interior grid coordinates)
    Xi = np.stack([gx[1:-1, 1:-1, 1:-1].ravel(),
                   gy[1:-1, 1:-1, 1:-1].ravel(),
                   gz[1:-1, 1:-1, 1:-1].ravel()], axis=1)
    ut = u_exact_np(Xi)
    rel_exact = float(np.linalg.norm(up - ut) / np.linalg.norm(ut))
    return rel_fd, rel_exact


# --------------------------------------------------------------------------- #
# Training loop.                                                              #
# --------------------------------------------------------------------------- #
def train(mode, lam, seed, steps, a):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP().to(DEV, DTYPE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-5)
    Xi, Xb = make_batch(seed)
    f_int = f_rhs_t(Xi, a)
    g_bnd = u_exact_t(Xb)
    hard = mode == "hard"
    t0 = time.time()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        _, Lu = anisotropic_laplacian(model, Xi, a, hard=hard)
        res = Lu - f_int
        loss = torch.mean(res ** 2)
        if not hard:
            u_b = model(Xb).squeeze(-1)
            loss = loss + lam * torch.mean((u_b - g_bnd) ** 2)
        loss.backward()
        opt.step()
        sched.step()
    torch.cuda.synchronize()
    wall = time.time() - t0
    return model, wall


# --------------------------------------------------------------------------- #
# Tangent-kernel constant measurement at the warmup point (kernel-side).      #
# Mirrors exp_prospective_trained.measure_constants, adapted to 3D anisotropy. #
# --------------------------------------------------------------------------- #
N_MEAS_INT, N_MEAS_BND = 260, 160     # held-out measurement rows (kept small)


def residual_row(model, X1, a):
    ax, ay, az = a
    x = X1.clone().requires_grad_(True)
    u = model(x).squeeze(-1)
    du = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    uxx = torch.autograd.grad(du[:, 0].sum(), x, create_graph=True)[0][:, 0]
    uyy = torch.autograd.grad(du[:, 1].sum(), x, create_graph=True)[0][:, 1]
    uzz = torch.autograd.grad(du[:, 2].sum(), x, create_graph=True)[0][:, 2]
    Lu = -(ax * uxx + ay * uyy + az * uzz)
    return (Lu - f_rhs_t(x, a))[0]


def residual_row_hard(model, X1, a):
    ax, ay, az = a
    x = X1.clone().requires_grad_(True)
    u = transfinite_lifting_3d(x) + bubble(x) * model(x).squeeze(-1)
    du = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    uxx = torch.autograd.grad(du[:, 0].sum(), x, create_graph=True)[0][:, 0]
    uyy = torch.autograd.grad(du[:, 1].sum(), x, create_graph=True)[0][:, 1]
    uzz = torch.autograd.grad(du[:, 2].sum(), x, create_graph=True)[0][:, 2]
    Lu = -(ax * uxx + ay * uyy + az * uzz)
    return (Lu - f_rhs_t(x, a))[0]


def boundary_row(model, X1):
    return (model(X1).squeeze(-1) - u_exact_t(X1))[0]


def jacobian_rows(model, X, fn):
    params = [p for p in model.parameters() if p.requires_grad]
    rows, vals = [], []
    for i in range(X.shape[0]):
        v = fn(model, X[i:i + 1])
        g = torch.autograd.grad(v, params, retain_graph=False, allow_unused=True)
        g = [gi if gi is not None else torch.zeros_like(p) for gi, p in zip(g, params)]
        rows.append(torch.cat([gi.reshape(-1) for gi in g]).detach())
        vals.append(float(v.item()))
    return torch.stack(rows), np.array(vals)


def kappa_from_gram(G, tol=1e-9):
    ev = np.linalg.eigvalsh(G)
    nz = ev[ev > ev[-1] * tol]
    return ev[-1] / nz[0]


def measure_constants(model, a):
    """Kernel-side tangent-kernel constants at the (frozen) warmup point."""
    gen = torch.Generator(device="cpu").manual_seed(777)
    Xi = torch.rand((N_MEAS_INT, 3), generator=gen).to(DEV, DTYPE)
    Xb = cube_boundary_points(gen, N_MEAS_BND).to(DEV, DTYPE)
    m, b = Xi.shape[0], Xb.shape[0]
    Jr, r0 = jacobian_rows(model, Xi, lambda mdl, x1: residual_row(mdl, x1, a))
    Jb, b0 = jacobian_rows(model, Xb, boundary_row)
    Jr = (Jr / math.sqrt(m)).cpu().numpy().astype(np.float64)
    Jb = (Jb / math.sqrt(b)).cpu().numpy().astype(np.float64)
    r0 = r0 / math.sqrt(m)
    b0 = b0 / math.sqrt(b)

    Krr, Kbb, Krb = Jr @ Jr.T, Jb @ Jb.T, Jr @ Jb.T
    lams = np.logspace(-1, 7, 33)
    kaps = []
    for lam in lams:
        s = math.sqrt(lam)
        G = np.block([[Krr, s * Krb], [s * Krb.T, lam * Kbb]])
        kaps.append(float(kappa_from_gram(G)))

    Kbb_inv = np.linalg.pinv(Kbb, rcond=1e-10)
    K_red = Krr - Krb @ Kbb_inv @ Krb.T
    kap_red = float(kappa_from_gram(K_red))
    evr, Ur = np.linalg.eigh(K_red)
    nz = evr > evr[-1] * 1e-10
    mu = float(evr[nz][0])
    Pr = Ur[:, nz] @ Ur[:, nz].T
    AYm = Krb @ Kbb_inv
    Nmat = AYm.T @ (np.eye(m) - Pr) @ AYm
    nu = float(np.linalg.norm(Nmat, 2))
    sigmaB = float(np.linalg.norm(Jb, 2))
    # shadow price rho (linearized constrained residual)
    dw_p, *_ = np.linalg.lstsq(Jb, -b0, rcond=None)
    BIG = 1e8
    Astack = np.vstack([Jr, math.sqrt(BIG) * Jb])
    ystack = np.concatenate([-(r0 + Jr @ dw_p), np.zeros(b)])
    w_corr, *_ = np.linalg.lstsq(Astack, ystack, rcond=None)
    dw = dw_p + w_corr
    r_star = r0 + Jr @ dw
    y_star, *_ = np.linalg.lstsq(Jb.T, -(Jr.T @ r_star), rcond=None)
    rho = float(np.linalg.norm(y_star))
    return dict(lams=lams.tolist(), kappas=kaps, rho=rho, mu=mu, nu=nu,
                kappa_red=kap_red, sigmaB=sigmaB)


def measure_hard_clock(model_hard, a):
    gen = torch.Generator(device="cpu").manual_seed(777)
    Xi = torch.rand((N_MEAS_INT, 3), generator=gen).to(DEV, DTYPE)
    m = Xi.shape[0]
    Jr, _ = jacobian_rows(model_hard, Xi, lambda mdl, x1: residual_row_hard(mdl, x1, a))
    Jr = (Jr / math.sqrt(m)).cpu().numpy().astype(np.float64)
    return float(kappa_from_gram(Jr @ Jr.T))


# --------------------------------------------------------------------------- #
# Per-problem prospective driver.                                             #
# --------------------------------------------------------------------------- #
def run_problem(tag, a, fd, record):
    print("\n" + "=" * 78)
    print(f"PROBLEM {tag}  (anisotropy a={a})")
    print("=" * 78)
    print(f"[FD reference] {fd['n']}^3 interior unknowns, nnz={fd['nnz']}, "
          f"sparse-direct solve {fd['wall_s']:.2f}s; discretization ref err vs "
          f"continuous u = {fd['ref_err']:.3e}")

    # ---- warmup, then freeze (measurement point) ----
    print(f"[1] warmup: {WARM_STEPS} Adam steps at lam={LAM_WARM:g}, seed 0 ...")
    warm, warm_wall = train("soft", LAM_WARM, 0, WARM_STEPS, a)
    warm_hard, _ = train("hard", None, 0, WARM_STEPS, a)
    rel_fd_w, rel_ex_w = real_rel_l2(warm, fd, hard=False)
    print(f"    warmup soft rel_l2 vs FD={rel_fd_w:.3e} ({warm_wall:.1f}s train)")

    # ---- measure constants ----
    print("[2] measuring tangent-kernel constants at the warmup point ...")
    tC = time.time()
    C = measure_constants(warm, a)
    kap_hard = measure_hard_clock(warm_hard, a)
    lams = np.array(C["lams"]); kaps = np.array(C["kappas"])
    rho, mu, nu, kap_red, sigmaB = C["rho"], C["mu"], C["nu"], C["kappa_red"], C["sigmaB"]
    print(f"    constants measured in {time.time() - tC:.1f}s")

    # ---- COMMIT predictions before the sweep ----
    lam_valley = float(lams[int(np.argmin(kaps))])
    kap_min = float(kaps.min())
    flat = kaps <= 1.10 * kap_min
    v_lo, v_hi = float(lams[flat].min()), float(lams[flat].max())
    product_pred = rho * sigmaB ** 2 / mu
    hard_should_win = kap_red < kap_min
    prediction = dict(
        lam_valley=lam_valley, valley_region=[v_lo, v_hi], kappa_eff_min=kap_min,
        kappa_eff_flat_ratio=float(kaps.max() / kap_min), product_limit=product_pred,
        kappa_red=kap_red, kappa_hard_tangent=kap_hard,
        hard_should_win_on_statics=bool(hard_should_win),
        rho=rho, mu=mu, nu=nu, sigmaB=sigmaB)
    print("\n" + "#" * 78)
    print("# PRE-REGISTERED PREDICTION (committed BEFORE any confirming train)")
    print("#" * 78)
    print(f"  constants: rho={rho:.4e}  mu={mu:.4e}  nu={nu:.4e}  "
          f"sigma_B={sigmaB:.4e}  kappa_red={kap_red:.4e}")
    print(f"  kappa_eff(lam) min = {kap_min:.4e} at lam = {lam_valley:g}")
    print(f"  --> PREDICTED conditioning-valley weight: lam = {lam_valley:g}")
    print(f"  --> PREDICTED best-accuracy region (flat floor): [{v_lo:g}, {v_hi:g}]")
    print(f"  --> PREDICTED product limit rho*sigma_B^2/mu = {product_pred:.4e}")
    print(f"  --> PREDICTED hard twin beats soft on statics: {hard_should_win} "
          f"(kappa_red={kap_red:.3e} vs best soft kappa_eff={kap_min:.3e})")
    print("#" * 78 + "\n")

    # persist the commitment immediately (before any sweep result exists)
    record[tag] = dict(anisotropy=list(a), fd_reference=dict(
        n=fd["n"], nnz=fd["nnz"], wall_s=fd["wall_s"], ref_err=fd["ref_err"]),
        warmup=dict(rel_l2_vs_fd=rel_fd_w, rel_l2_vs_exact=rel_ex_w, wall_s=warm_wall),
        prediction=prediction, constants=C, status="prediction_committed")
    OUT.write_text(json.dumps(record, indent=1))

    # ---- confirming sweep (soft) ----
    print(f"[3] confirming soft sweep: lams={SWEEP_LAMS}, {SWEEP_SEEDS} seeds, "
          f"{SWEEP_STEPS} steps each ...")
    sweep = {}
    for lam in SWEEP_LAMS:
        rel_fds, rel_exs, walls = [], [], []
        for seed in range(SWEEP_SEEDS):
            m, w = train("soft", lam, seed, SWEEP_STEPS, a)
            rfd, rex = real_rel_l2(m, fd, hard=False)
            rel_fds.append(rfd); rel_exs.append(rex); walls.append(w)
        sweep[f"{lam:g}"] = dict(
            rel_l2_vs_fd=rel_fds, rel_l2_vs_exact=rel_exs, wall_s=walls,
            median_rel_l2_vs_fd=float(np.median(rel_fds)),
            median_rel_l2_vs_exact=float(np.median(rel_exs)),
            median_wall_s=float(np.median(walls)))
        print(f"    soft lam={lam:8g}  median rel_l2(FD)={np.median(rel_fds):.3e}  "
              f"median rel_l2(exact)={np.median(rel_exs):.3e}  "
              f"({np.median(walls):.1f}s/train)")
        record[tag]["sweep"] = sweep
        OUT.write_text(json.dumps(record, indent=1))

    # ---- hard twin at matched budget (extra seeds to expose fragility) ----
    print(f"[4] hard twin: {HARD_SEEDS} seeds, {SWEEP_STEPS} steps each ...")
    hard_fds, hard_exs, hard_walls = [], [], []
    for seed in range(HARD_SEEDS):
        m, w = train("hard", None, seed, SWEEP_STEPS, a)
        rfd, rex = real_rel_l2(m, fd, hard=True)
        hard_fds.append(rfd); hard_exs.append(rex); hard_walls.append(w)
        print(f"    hard seed={seed}  rel_l2(FD)={rfd:.3e}  rel_l2(exact)={rex:.3e}  "
              f"({w:.1f}s)")
    hard_med = float(np.median(hard_fds))
    # basin-escape fragility: fraction of seeds that failed (rel_l2 > 5x median)
    hard_arr = np.array(hard_fds)
    frac_fragile = float(np.mean(hard_arr > 5.0 * np.median(hard_arr)))

    # ---- score ----
    soft_meds = np.array([sweep[f"{l:g}"]["median_rel_l2_vs_fd"] for l in SWEEP_LAMS])
    i_best = int(np.argmin(soft_meds))
    lam_best_obs = float(SWEEP_LAMS[i_best])
    best_soft_med = float(soft_meds[i_best])
    valley_depth = float(soft_meds.max() / soft_meds.min())     # accuracy U depth
    obs_region = [SWEEP_LAMS[i] for i in range(len(SWEEP_LAMS))
                  if soft_meds[i] <= 2.0 * best_soft_med]
    region_overlap = [L for L in obs_region if v_lo <= L <= v_hi]
    hard_advantage = best_soft_med / hard_med
    hard_won = hard_med < best_soft_med
    hard_pred_correct = (hard_should_win == hard_won)

    confirmation = dict(
        lam_best_observed=lam_best_obs, best_soft_median_vs_fd=best_soft_med,
        accuracy_valley_depth_x=valley_depth, observed_best_region=obs_region,
        predicted_valley_region=[v_lo, v_hi],
        predicted_valley_in_observed_region=bool(region_overlap),
        hard_median_vs_fd=hard_med, hard_advantage_x=hard_advantage,
        hard_won_observed=hard_won, hard_prediction_correct=hard_pred_correct,
        hard_fragile_fraction=frac_fragile)
    record[tag].update(dict(
        hard=dict(rel_l2_vs_fd=hard_fds, rel_l2_vs_exact=hard_exs, wall_s=hard_walls,
                  median_rel_l2_vs_fd=hard_med, fragile_fraction=frac_fragile),
        confirmation=confirmation, status="confirmed"))
    OUT.write_text(json.dumps(record, indent=1))

    print("\n" + "-" * 78)
    print(f"SCORING  problem {tag}")
    print("-" * 78)
    print(f"  soft accuracy valley: best rel_l2(FD)={best_soft_med:.3e} at lam={lam_best_obs:g}; "
          f"valley depth {valley_depth:.1f}x across the sweep")
    print(f"  predicted valley weight lam={lam_valley:g}; predicted region [{v_lo:g},{v_hi:g}]; "
          f"observed best region {obs_region}")
    print(f"  predicted region overlaps observed best region: {bool(region_overlap)} "
          f"(overlap={region_overlap})")
    print(f"  hard twin median rel_l2(FD)={hard_med:.3e}  advantage over best soft "
          f"{hard_advantage:.2f}x  (won={hard_won})")
    print(f"  hard-win prediction correct: {hard_pred_correct}  "
          f"(predicted={hard_should_win}, observed={hard_won})")
    print(f"  hard basin-escape fragility: {frac_fragile:.0%} of {HARD_SEEDS} seeds failed")
    print("-" * 78)
    return record[tag]


def main():
    t0 = time.time()
    assert torch.cuda.is_available(), "CUDA not available; task requires the GPU"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("=" * 78)
    print("LARGE TRAINED 3D PINN DIAGNOSTIC ON THE GPU")
    print(f"device={DEV} ({torch.cuda.get_device_name(0)}), net={DEPTH}x{WIDTH} tanh, "
          f"N_int={N_INT}, N_bnd={N_BND}")
    print(f"sweep lams={SWEEP_LAMS}, soft seeds={SWEEP_SEEDS}, hard seeds={HARD_SEEDS}, "
          f"steps={SWEEP_STEPS}")
    print("finite-difference solve: reference field and timing context")
    print("=" * 78)

    record = dict(
        description=("Large trained 3D PINN boundary-weight sweep with a hard-constrained "
                     "twin on 3D Poisson and a 3D stiff-anisotropic elliptic problem. A "
                     "conventional 7-point finite-difference sparse-direct solve provides "
                     "reference truth for a real relative-L2 error and timing CONTEXT ONLY "
                     "(not a speed comparison). Tangent-kernel constants and predictions "
                     "are committed before the confirming sweep."),
        config=dict(width=WIDTH, depth=DEPTH, n_int=N_INT, n_bnd=N_BND,
                    warm_steps=WARM_STEPS, sweep_steps=SWEEP_STEPS, lam_warm=LAM_WARM,
                    sweep_lams=SWEEP_LAMS, sweep_seeds=SWEEP_SEEDS, hard_seeds=HARD_SEEDS,
                    fd_n=FD_N, frequencies=[KX, KY, KZ], device=torch.cuda.get_device_name(0)),
        timing_disclaimer="Finite-difference wall times are reference context.")

    # finite-difference references first (reference truth for both problems)
    print("\n[FD] assembling and solving the two finite-difference references ...")
    fd_p = fd_reference(A_POISSON)
    print(f"  Poisson FD: {fd_p['wall_s']:.2f}s, ref err {fd_p['ref_err']:.3e}")
    fd_s = fd_reference(A_STIFF)
    print(f"  stiff   FD: {fd_s['wall_s']:.2f}s, ref err {fd_s['ref_err']:.3e}")

    run_problem("P1_poisson", A_POISSON, fd_p, record)
    run_problem("P2_stiff", A_STIFF, fd_s, record)

    record["total_wall_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(record, indent=1))

    # ---- final summary table ----
    print("\n" + "=" * 78)
    print("SUMMARY (per problem)")
    print("=" * 78)
    for tag in ("P1_poisson", "P2_stiff"):
        r = record[tag]
        c = r["confirmation"]
        p = r["prediction"]
        print(f"\n{tag} (a={r['anisotropy']}):")
        print(f"  FD reference: {r['fd_reference']['n']}^3, {r['fd_reference']['wall_s']:.2f}s, "
              f"disc.err {r['fd_reference']['ref_err']:.2e}")
        print(f"  soft valley: best rel_l2(FD)={c['best_soft_median_vs_fd']:.3e} at "
              f"lam={c['lam_best_observed']:g}, depth {c['accuracy_valley_depth_x']:.1f}x")
        print(f"  predicted valley lam={p['lam_valley']:g}; region {p['valley_region']}; "
              f"overlaps observed: {c['predicted_valley_in_observed_region']}")
        print(f"  hard twin rel_l2(FD)={c['hard_median_vs_fd']:.3e}, "
              f"advantage {c['hard_advantage_x']:.2f}x, fragile {c['hard_fragile_fraction']:.0%}, "
              f"hard-win pred correct: {c['hard_prediction_correct']}")
        print(f"  product limit rho*sigmaB^2/mu = {p['product_limit']:.3e}, kappa_red={p['kappa_red']:.3e}")
    print(f"\nwrote {OUT}")
    print(f"TOTAL WALL: {record['total_wall_s']}s")


if __name__ == "__main__":
    main()
