"""Shared problem builders for the penalty-tax verifications.

Linear-model PINN discretizations of Poisson problems. The model class is a
feature map phi: R^d -> R^n; the interior operator applies L = -Laplacian to
the features at collocation points, the boundary operator evaluates the trace.

Two feature families:
- Fourier features in 1D (odd/even sines and cosines): smooth, well understood.
- Random Fourier features in 2D (Gaussian kernel approximation).

Underparameterized (n < m) => misfit regime (shadow price rho > 0).
Overparameterized (n > m + b) => realizable regime (rho ~ 0, U-shaped kappa).
"""

import numpy as np


def fourier_features_1d(x, n_feat, ell=1.0, lifting=True):
    """phi_j(x) = sin(j pi x / ell) and cos(j pi x / ell), optionally + {1, x}.

    With lifting=True the basis contains the 1D harmonic functions {1, x}
    (zero Laplacian, boundary traces spanning R^2): a FREE LIFTING, which makes
    the boundary constraint costless (shadow price zero at every weight). With
    lifting=False the model has no zero-residual escape directions and the
    constraint genuinely competes with the interior fit.

    Returns (Phi, Lap): feature matrix and its Laplacian (second derivative),
    both (len(x), n_feat).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    cols, lap_cols = [], []
    if lifting:
        # constant and linear (harmonic in 1D: second derivative zero)
        cols.append(np.ones_like(x));     lap_cols.append(np.zeros_like(x))
        cols.append(x);                   lap_cols.append(np.zeros_like(x))
    j = 1
    while len(cols) < n_feat:
        w = j * np.pi / ell
        cols.append(np.sin(w * x));       lap_cols.append(-(w ** 2) * np.sin(w * x))
        if len(cols) < n_feat:
            cols.append(np.cos(w * x));   lap_cols.append(-(w ** 2) * np.cos(w * x))
        j += 1
    Phi = np.stack(cols, axis=1)
    Lap = np.stack(lap_cols, axis=1)
    return Phi, Lap


def build_poisson_1d(n_feat=24, m=200, b=2, freq=2.5, seed=0, lifting=False,
                     bump=True):
    """-u'' = f on (0,1), Dirichlet data, manufactured non-representable target.

    u(x) = sin(freq*pi*x) + bump term. A non-integer freq and the Gaussian bump
    both put the target outside the integer-harmonic span (misfit, rho > 0
    when lifting=False). With lifting=True the basis gains {1, x} and the
    shadow price collapses to zero (free lifting), whatever the misfit.
    """
    rng = np.random.default_rng(seed)
    xs = np.sort(rng.uniform(0.0, 1.0, size=m))

    def u_exact(x):
        out = np.sin(freq * np.pi * x)
        if bump:
            out = out + 0.5 * np.exp(-((x - 0.35) / 0.12) ** 2)
        return out

    def f_fun(x):  # f = -u''
        out = (freq * np.pi) ** 2 * np.sin(freq * np.pi * x)
        if bump:
            s = (x - 0.35) / 0.12
            out = out + 0.5 * np.exp(-s ** 2) * (2.0 / 0.12 ** 2) * (1.0 - 2.0 * s ** 2)
        return out

    Phi_i, Lap_i = fourier_features_1d(xs, n_feat, lifting=lifting)
    A = -Lap_i / np.sqrt(m)          # residual operator for L = -d^2/dx^2
    f = f_fun(xs) / np.sqrt(m)
    sb = np.array([0.0, 1.0])
    Phi_b, _ = fourier_features_1d(sb, n_feat, lifting=lifting)
    B = Phi_b / np.sqrt(b)
    g = u_exact(sb) / np.sqrt(b)
    return dict(A=A, f=f, B=B, g=g, x=xs, u_exact=u_exact, n=n_feat, m=m, b=b)


def shifted_sine_features_1d(x, n_feat, delta=0.15):
    """Dirichlet eigenfunctions of an EXTENDED interval, restricted to (0,1).

    phi_j(x) = sin(j pi (x + delta) / (1 + 2 delta)): each is a Laplacian
    eigenfunction with eigenvalue (j pi / (1+2 delta))^2 > 0, nonvanishing at
    x = 0, 1. The span contains no near-harmonic direction (the operator is
    diagonal in this basis with eigenvalues bounded away from zero), so the
    interior operator stays well-conditioned and float64 can resolve the
    statics exactly. No free lifting.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    ell = 1.0 + 2.0 * delta
    cols, lap_cols = [], []
    for j in range(1, n_feat + 1):
        w = j * np.pi / ell
        phi = np.sin(w * (x + delta))
        cols.append(phi)
        lap_cols.append(-(w ** 2) * phi)
    return np.stack(cols, axis=1), np.stack(lap_cols, axis=1)


def build_poisson_1d_eig(n_feat=24, m=200, freq=2.5, delta=0.15, seed=0,
                         normalize=False):
    """Misfit 1D Poisson in the eigenfunction basis: clean statics testbed.

    With normalize=True each feature is rescaled so its interior column has
    unit norm (Jacobi preconditioning of A). This removes the representation
    and operator conditioning axes and leaves ONLY the constraint-imposition
    conditioning: the best case for a soft penalty. The leak and the shadow
    price are invariant to this reparameterization; the curvature constants
    are not (they are the dynamics side of the story).
    """
    rng = np.random.default_rng(seed)
    xs = np.sort(rng.uniform(0.0, 1.0, size=m))

    def u_exact(x):
        return np.sin(freq * np.pi * x) + 0.5 * np.exp(-((x - 0.35) / 0.12) ** 2)

    def f_fun(x):  # f = -u''
        s = (x - 0.35) / 0.12
        return ((freq * np.pi) ** 2 * np.sin(freq * np.pi * x)
                + 0.5 * np.exp(-s ** 2) * (2.0 / 0.12 ** 2) * (1.0 - 2.0 * s ** 2))

    Phi_i, Lap_i = shifted_sine_features_1d(xs, n_feat, delta)
    A = -Lap_i / np.sqrt(m)
    f = f_fun(xs) / np.sqrt(m)
    sb = np.array([0.0, 1.0])
    Phi_b, _ = shifted_sine_features_1d(sb, n_feat, delta)
    b = 2
    B = Phi_b / np.sqrt(b)
    g = u_exact(sb) / np.sqrt(b)
    if normalize:
        s = np.linalg.norm(A, axis=0)
        A = A / s[None, :]
        B = B / s[None, :]
    return dict(A=A, f=f, B=B, g=g, x=xs, u_exact=u_exact, n=n_feat, m=m, b=b)


def tensor_fourier_2d(X, J):
    """Well-conditioned 2D basis: products of {sin,cos}(j pi x) x {sin,cos}(k pi y).

    Returns (Phi, Lap) with Lap the (negative-definite) Laplacian of each
    feature. No harmonic completion: every feature has nonzero Laplacian.
    """
    x, y = X[:, 0], X[:, 1]
    cols, laps = [], []
    for j in range(1, J + 1):
        for k in range(1, J + 1):
            wj, wk = j * np.pi, k * np.pi
            fac = -(wj ** 2 + wk ** 2)
            for fx, gx in ((np.sin, np.sin), (np.sin, np.cos),
                           (np.cos, np.sin), (np.cos, np.cos)):
                phi = fx(wj * x) * gx(wk * y)
                cols.append(phi)
                laps.append(fac * phi)
    Phi = np.stack(cols, axis=1)
    Lap = np.stack(laps, axis=1)
    return Phi, Lap


def build_poisson_2d(J=5, m=1200, b=12, kx=2.5, ky=1.5, seed=0):
    """-Lap u = f on the unit square, Dirichlet data, u = sin(kx pi x) sin(ky pi y).

    Tensor Fourier model (n = 4 J^2 features), well-conditioned. Non-integer
    kx, ky put the target outside the span: misfit regime with rho > 0. The
    basis has no zero-Laplacian direction, so there is no free lifting. b is
    kept modest so that B B^T stays well-conditioned.
    """
    rng = np.random.default_rng(seed)
    Xi = rng.uniform(0.0, 1.0, size=(m, 2))
    Phi_i, Lap_i = tensor_fourier_2d(Xi, J)
    A = -Lap_i / np.sqrt(m)
    u_ex = lambda X: np.sin(kx * np.pi * X[:, 0]) * np.sin(ky * np.pi * X[:, 1])
    lam_ex = (kx ** 2 + ky ** 2) * np.pi ** 2
    f = lam_ex * u_ex(Xi) / np.sqrt(m)
    # boundary points spread over the four edges
    t = rng.uniform(0.0, 1.0, size=b)
    edge = np.arange(b) % 4
    Xb = np.zeros((b, 2))
    Xb[:, 0] = np.where(edge == 0, t, np.where(edge == 1, t, np.where(edge == 2, 0.0, 1.0)))
    Xb[:, 1] = np.where(edge == 0, 0.0, np.where(edge == 1, 1.0, np.where(edge == 2, t, t)))
    Phi_b, _ = tensor_fourier_2d(Xb, J)
    B = Phi_b / np.sqrt(b)
    g = u_ex(Xb) / np.sqrt(b)
    return dict(A=A, f=f, B=B, g=g, Xi=Xi, Xb=Xb, u_exact=u_ex, n=Phi_i.shape[1],
                m=m, b=b)


def constrained_solution(A, f, B, g):
    """Solve min ||Aw-f||^2 s.t. Bw=g via nullspace method; return w*, y*, r*.

    KKT: A^T r* + B^T y* = 0 with r* = A w* - f; y* = -(B B^T)^{-1} B A^T r*.
    """
    n = A.shape[1]
    # particular solution of Bw=g
    w_p = np.linalg.lstsq(B, g, rcond=None)[0]
    # nullspace basis of B
    _, s, Vt = np.linalg.svd(B, full_matrices=True)
    rank = int(np.sum(s > s[0] * 1e-12)) if s.size else 0
    Z = Vt[rank:].T                      # n x (n - rank)
    v = np.linalg.lstsq(A @ Z, f - A @ w_p, rcond=None)[0]
    w_star = w_p + Z @ v
    r_star = A @ w_star - f
    # stable multiplier: least-squares solve of B^T y = -A^T r*
    y_star = np.linalg.lstsq(B.T, -(A.T @ r_star), rcond=None)[0]
    return w_star, y_star, r_star, Z


def penalized_solution(A, f, B, g, lam):
    """Minimizer of the soft objective at weight lam."""
    H = A.T @ A + lam * (B.T @ B)
    rhs = A.T @ f + lam * (B.T @ g)
    return np.linalg.solve(H, rhs), H
