"""Numerical validation of the exact penalty-method laws (fixed-feature regime).

Each test builds a collocated linear PINN discretization (A w ~ f interior,
B w ~ g boundary), solves the hard-constrained least-squares problem for the
multiplier y*, and checks a law that must hold to round-off at every weight.
Runs in a few seconds on a laptop CPU.
"""
import numpy as np
import pytest

from problems import (build_poisson_1d_eig, build_poisson_2d,
                      constrained_solution, penalized_solution)
from verify_exact_laws import schur_N

PROBLEMS = {
    "1d_eig": lambda: build_poisson_1d_eig(normalize=True),
    "2d_tensor": lambda: build_poisson_2d(),
}
LAMS = np.logspace(-2, 8, 31)


def constants(prob):
    A, f, B, g = prob["A"], prob["f"], prob["B"], prob["g"]
    w_star, y_star, r_star, Z = constrained_solution(A, f, B, g)
    N = schur_N(A, B, Z)
    mu = np.linalg.eigvalsh(Z.T @ (A.T @ A) @ Z)[0]
    return dict(A=A, f=f, B=B, g=g, w_star=w_star, y_star=y_star, N=N,
                rho=np.linalg.norm(y_star), nu=np.linalg.norm(N, 2), mu=mu,
                sB2=np.linalg.norm(B, 2) ** 2)


@pytest.fixture(params=sorted(PROBLEMS), scope="module")
def c(request):
    return constants(PROBLEMS[request.param]())


def test_kkt_of_hard_problem(c):
    """A^T r* + B^T y* = 0 and B w* = g at the constrained optimum."""
    A, f, B, g = c["A"], c["f"], c["B"], c["g"]
    r = A @ c["w_star"] - f
    assert np.linalg.norm(A.T @ r + B.T @ c["y_star"]) < 1e-9 * np.linalg.norm(A.T @ f)
    assert np.linalg.norm(B @ c["w_star"] - g) < 1e-10 * max(1.0, np.linalg.norm(g))


def test_resolvent_identity(c):
    """Boundary residual of the penalized minimizer is exactly (N + lam I)^{-1} y*."""
    A, f, B, g = c["A"], c["f"], c["B"], c["g"]
    I = np.eye(B.shape[0])
    for lam in (1e-1, 1e1, 1e3, 1e5):
        w, _ = penalized_solution(A, f, B, g, lam)
        q = B @ w - g
        pred = -np.linalg.solve(c["N"] + lam * I, c["y_star"])
        rel = min(np.linalg.norm(q - pred), np.linalg.norm(q + pred)) / np.linalg.norm(q)
        assert rel < 1e-6, (lam, rel)


def test_two_sided_leak_bracket(c):
    """rho/(lam+nu) <= ||B w_lam - g|| <= rho/lam at every weight, strictly decreasing."""
    A, f, B, g = c["A"], c["f"], c["B"], c["g"]
    leaks = np.array([np.linalg.norm(B @ penalized_solution(A, f, B, g, l)[0] - g)
                      for l in LAMS])
    assert np.all(leaks <= c["rho"] / LAMS * (1 + 1e-6))
    assert np.all(leaks >= c["rho"] / (LAMS + c["nu"]) * (1 - 1e-6))
    assert np.all(np.diff(leaks) < 0)


def test_multiplier_estimate_converges_to_shadow_price(c):
    """lam * leak rises to rho and never overshoots it."""
    A, f, B, g = c["A"], c["f"], c["B"], c["g"]
    y = np.array([l * np.linalg.norm(B @ penalized_solution(A, f, B, g, l)[0] - g)
                  for l in LAMS])
    assert np.all(y <= c["rho"] * (1 + 1e-6))
    assert abs(y[-1] / c["rho"] - 1) < 1e-4


@pytest.mark.parametrize("lam0", [1e0, 1e2, 1e4])
def test_pareto_slope_and_envelope(c, lam0):
    """dP/dQ = -lam along the path, and J'(lam) = Q/2 (envelope law)."""
    A, f, B, g = c["A"], c["f"], c["B"], c["g"]

    def vals(lam):
        w, _ = penalized_solution(A, f, B, g, lam)
        P = np.linalg.norm(A @ w - f) ** 2
        Q = np.linalg.norm(B @ w - g) ** 2
        return P, Q, 0.5 * P + 0.5 * lam * Q

    h = 1e-3 * lam0
    Pp, Qp, Jp = vals(lam0 + h)
    Pm, Qm, Jm = vals(lam0 - h)
    _, Q0, _ = vals(lam0)
    assert abs((Pp - Pm) / (Qp - Qm) + lam0) / lam0 < 1e-4
    assert abs((Jp - Jm) / (2 * h) - 0.5 * Q0) / (0.5 * Q0) < 1e-4


def test_leak_conditioning_product(c):
    """leak * kappa(H_lam) >= rho sB^2 / (2 mu) for lam >= nu, and -> rho sB^2 / mu."""
    A, f, B, g = c["A"], c["f"], c["B"], c["g"]
    bound = c["rho"] * c["sB2"] / c["mu"]
    lams = np.concatenate([LAMS[LAMS >= c["nu"]], [1e8]])
    prods = []
    for lam in lams:
        w, H = penalized_solution(A, f, B, g, lam)
        ev = np.linalg.eigvalsh(H)
        prods.append(np.linalg.norm(B @ w - g) * ev[-1] / ev[0])
    prods = np.array(prods)
    assert np.all(prods >= 0.5 * bound * (1 - 1e-6))
    assert abs(prods[-1] / bound - 1) < 1e-2          # large-weight limit, 1%


def test_fence_bound():
    """No weight conditions the penalized Hessian below the reduced (hard) problem."""
    p = build_poisson_1d_eig(normalize=True)
    A, f, B, g = p["A"], p["f"], p["B"], p["g"]
    _, _, _, Z = constrained_solution(A, f, B, g)
    ev = np.linalg.eigvalsh(Z.T @ (A.T @ A) @ Z)
    kred = ev[-1] / ev[0]
    for lam in np.logspace(-2, 8, 41):
        _, H = penalized_solution(A, f, B, g, lam)
        e = np.linalg.eigvalsh(H)
        assert e[-1] / e[0] >= kred * (1 - 1e-8)


def test_momentum_family_slow_mode_floor():
    """Every stable constant heavy-ball pair keeps the slow mode above 1 - sqrt(eta h_s)."""
    for beta in np.linspace(0.0, 0.999, 40):
        for x in np.linspace(1e-4, 1.0, 40):        # x = eta * h_s
            r = np.max(np.abs(np.roots([1.0, -(1 + beta - x), beta])))
            assert r >= 1 - np.sqrt(x) - 1e-12


def test_free_lifting_has_zero_shadow_price():
    """A basis containing a harmonic lifting pays nothing for the constraint."""
    from problems import build_poisson_1d
    p = build_poisson_1d(lifting=True)
    A, f, B, g = p["A"], p["f"], p["B"], p["g"]
    _, y_star, _, _ = constrained_solution(A, f, B, g)
    assert np.linalg.norm(y_star) < 1e-8 * np.linalg.norm(A.T @ f)
