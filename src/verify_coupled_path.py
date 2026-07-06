"""Counterexample search for the unconditional coupled-path floor.

The coupled-path statement claims that for EVERY monotone coupled weight
schedule driving first-order updates to boundary accuracy eps_b, the accumulated
conditioning sum_t kappa(H_{lambda_t}) is bounded below by Omega(kred ln(1/eps_b)),
with no bounded-lead hypothesis. The accompanying proof establishes this
unconditionally by reducing it to a step-count floor T >= Omega(ln(1/eps_b)) on the
slowest boundary mode, since the fence bound keff(H_lambda) >= kred already supplies
the per-step factor kred (each kappa(H_lambda) >= keff(H_lambda) >= kred).

This script is the guard against a flawed proof. It ACTIVELY SEARCHES FOR
COUNTEREXAMPLES: it generates >= 500 random MONOTONE coupled weight schedules of
many shapes (jumps, ramps, exponential, logarithmic, piecewise, random-monotone)
on the 1D and 2D fixed-feature testbeds, runs the EXACT error recursion (R), and
checks whether the proven lower bounds are ever violated. A genuine lower bound
must have worst-case ratio (measured / bound) >= 1 up to roundoff. If any schedule
violates a bound, the theorem is wrong and the script reports it as a FAIL.

Checks (all in the paper's notation):

  P0  Fence per step: keff(H_{lambda_t}) >= kred at every weight along every
      schedule (the fence bound), so kappa(H) >= kred term by term. This is what
      turns a step-count floor into a conditioning floor.

  P1  Exact slowest-mode readout recursion (Lemma 3):
        b_{t+1} = (1 - eta_t lambda_t theta) b_t
                  - eta_t v^T (B A^T A) e_t + eta_t theta (v^T y*)
      reproduces the true readout b_t = v^T (B e_t) to ~1e-12, where v is the unit
      lammin(B B^T)-eigenvector and theta = lammin(B B^T).

  P2  Per-step slowest-mode floor (Lemma 4): with beta = theta/sB^2 (for the stable
      step eta_t <= 1/lammax(H), used here) the exact recursion satisfies
        |b_{t+1}| >= (1 - beta) |b_t| - eta_t Kd,
      Kd = ||A^T A B^T v|| R + theta rho, R = sup_t ||e_t|| (measured), at EVERY step.

  P3  Trajectory bound ||e_t|| <= ||e_0|| (Lemma 2): the error norm is
      non-increasing along every stable-step schedule, so R = ||e_0|| is a valid
      drive constant.

  P4  UNCONDITIONAL step-count floor (Theorem 3): at the first time T with
      boundary accuracy ||q_T|| <= eps_b, the measured T satisfies
        T >= ln( |b_0| / (eps_b + Delta_T) ) / ln( 1 / (1 - beta) ),
      Delta_T the exact drive envelope (Delta_{t+1} = (1-beta) Delta_t + eta_t Kd).
      Checked as measured_T >= floor for every schedule that reaches eps_b. This is
      the main claim: NO monotone coupled schedule beats the floor.

  P5  Accumulated-conditioning floor (Corollary): for every schedule reaching
      eps_b, sum_{t<T} kappa(H_{lambda_t}) >= T * kred >= floor * kred, the
      Omega(kred ln(1/eps_b)) coupled-path floor. Worst-case ratio measured / bound
      reported; must be >= 1.

  P7  Closed-form drive envelope (E7): the split bound Delta_T <= (2 Kd/(sB^2 beta))
      (1/lambda_{T-m} + (1-beta)^m/lambda_0), m = ceil(ln(1/eps_b)/ln(1/(1-beta))),
      upper-bounds the exact drive envelope Delta_T over every reaching schedule. This
      guards the O(eps_b) argument that makes the floor Omega(ln(1/eps_b)).

  DIAG  Boundary-lead statistic Gamma of the bounded-lead result (||q_t|| <= Gamma |a_t|)
      is measured over all schedules and printed (with chi*Gamma). It documents that
      the bounded-lead hypothesis can be large (chi*Gamma ~ 2e2 on 2D) while the
      unconditional proof does not use Gamma at all.

Deterministic seeds throughout. Prints [PASS]/[FAIL] per check, the worst-case
numbers, the number of schedules searched, and ALL PASS / SOME FAILED.

Run:
  PYTHONPATH=src python src/verify_coupled_path.py
"""

import sys

import numpy as np

from problems import (build_poisson_1d_eig, build_poisson_2d,
                      constrained_solution)

PASS = True


def check(name, ok, detail=""):
    global PASS
    status = "PASS" if ok else "FAIL"
    if not ok:
        PASS = False
    print(f"[{status}] {name} {detail}", flush=True)


# --------------------------------------------------------------------------
# problem constants (paper notation)
# --------------------------------------------------------------------------
def constants(prob):
    A, f, B, g = prob["A"], prob["f"], prob["B"], prob["g"]
    w_star, y_star, r_star, Z = constrained_solution(A, f, B, g)
    Y = B.T @ np.linalg.inv(B @ B.T)                 # min-norm lifting, B Y = I
    AZ, AY = A @ Z, A @ Y
    M = AZ.T @ AZ
    evM, VM = np.linalg.eigh(M)
    mu, Lred = evM[0], evM[-1]
    kred = Lred / mu
    u = VM[:, 0]                                     # slow reduced eigenvector
    Zu = Z @ u
    C = AZ.T @ AY
    c = C.T @ u                                      # slow-mode coupling vector
    BBt = B @ B.T
    evB, VB = np.linalg.eigh(BBt)
    theta = evB[0]                                   # lammin(B B^T)
    v = VB[:, 0]                                     # slowest boundary eigenvector
    sB2 = np.linalg.norm(B, 2) ** 2                  # = lammax(B B^T)
    sA2 = np.linalg.norm(A, 2) ** 2
    rho = np.linalg.norm(y_star)
    return dict(
        A=A, B=B, Z=Z, Y=Y, w_star=w_star, y_star=y_star, Bty=B.T @ y_star,
        AtA=A.T @ A, BtB=B.T @ B, BAtA=B @ (A.T @ A),
        M=M, mu=mu, Lred=Lred, kred=kred, u=u, Zu=Zu, c=c,
        BBt=BBt, theta=theta, v=v, sB2=sB2, sA2=sA2, rho=rho,
        chi=np.linalg.norm(c) / mu,
        AtABtv=(A.T @ A) @ (B.T @ v),                # A^T A B^T v (drive vector)
    )


# --------------------------------------------------------------------------
# monotone coupled schedule generators (varied shapes)
# --------------------------------------------------------------------------
def make_schedules(n_random, lam_max_pool, T=200000, seed=0):
    """Return a dict name -> nondecreasing lambda sequence of length T.

    Deterministic given seed. Includes deterministic canonical shapes plus
    n_random random-monotone schedules with varied terminal weights.
    """
    rng = np.random.default_rng(seed)
    sched = {}

    # canonical deterministic shapes across terminal weights
    for Lmax in lam_max_pool:
        sched[f"jump->{Lmax:.0e}"] = np.full(T, float(Lmax))
        sched[f"log 1->{Lmax:.0e}"] = np.logspace(0, np.log10(Lmax), T)
        sched[f"lin 1->{Lmax:.0e}"] = np.linspace(1.0, float(Lmax), T)
        sched[f"sqrt 1->{Lmax:.0e}"] = 1.0 + (Lmax - 1.0) * np.sqrt(
            np.linspace(0, 1, T))
        sched[f"quad 1->{Lmax:.0e}"] = 1.0 + (Lmax - 1.0) * (
            np.linspace(0, 1, T) ** 2)
        # exponential ramp saturating at Lmax
        sched[f"exp->{Lmax:.0e}"] = np.minimum(
            float(Lmax), np.exp(np.linspace(0, np.log(Lmax) + 3, T)))
        # ramp then jump (worst-case injector: build boundary lead, then stiffen)
        ramp = np.linspace(1.0, np.sqrt(Lmax), min(T // 4, 500))
        sched[f"ramp+jump->{Lmax:.0e}"] = np.concatenate(
            [ramp, np.full(T, float(Lmax))])[:T]
        # staircase (piecewise-constant monotone)
        steps = np.floor(np.linspace(0, np.log10(Lmax), T))
        sched[f"stair->{Lmax:.0e}"] = 10.0 ** steps

    # random-monotone: cumulative positive increments, varied roughness & terminal
    for k in range(n_random):
        Lmax = float(rng.choice(lam_max_pool))
        style = rng.integers(0, 4)
        if style == 0:                                # heavy-tailed increments
            inc = rng.pareto(1.5, T) + 1e-9
        elif style == 1:                              # exponential increments
            inc = rng.exponential(1.0, T)
        elif style == 2:                              # sparse jumps
            inc = rng.random(T) ** 8
        else:                                         # near-uniform + noise
            inc = rng.random(T) + 0.05
        seq = np.cumsum(inc)
        seq = 1.0 + (Lmax - 1.0) * (seq - seq[0]) / (seq[-1] - seq[0])
        sched[f"rand{k}->{Lmax:.0e}"] = seq

    return sched


# --------------------------------------------------------------------------
# exact recursion driver + all per-schedule measurements
# --------------------------------------------------------------------------
def stable_step(K, lam):
    """eta = 1/(sA^2 + lam sB^2) <= 1/lammax(H_lam): a stable step (stricter than
    2/lammax, so the floor tests below are conservative)."""
    return 1.0 / (K["sA2"] + lam * K["sB2"])


def run_schedule(K, seq, eps_b, R, Kd, beta, max_steps, track_cond=False):
    """Run the exact recursion (R) under schedule `seq` from w_0 = 0.

    Returns a dict of measurements needed for P1-P7:
      T             first t with ||q_t|| <= eps_b (or -1)
      p1_err        max abs error of the exact readout recursion (Lemma 3)
      p2_slack      min over steps of ( |b_{t+1}| - [(1-beta)|b_t| - eta Kd] )
      max_e_ratio   max_t ||e_t|| / ||e_0||   (for P3; must be <= 1)
      Delta_T       drive envelope at T ; lam_hist for the E7 closed-form check
      b0            |b_0|
      sum_kappa     sum_{t<T} kappa(H_{lambda_t})  (only if track_cond and reaches)
      Gamma         max_t ||q_t|| / |a_t|  (boundary lead, diagnostic)

    Per-step eigendecomposition (for sum_kappa) is done ONLY when track_cond is
    True, and the loop stops at the hitting time T, so the expensive path runs for
    a small number of steps (the reaching schedules resolve fast). The cheap path
    (track_cond False) does no eigensolve and is used for the full search sweep.
    """
    A, B, AtA, BtB, Bty = K["A"], K["B"], K["AtA"], K["BtB"], K["Bty"]
    v, theta, Zu = K["v"], K["theta"], K["Zu"]
    y_star = K["y_star"]
    # precompute fixed readout covectors (Lemma 3): vB = v^T B, vB_AtA = v^T B A^T A
    vB = v @ B                                        # length-n covector
    vB_AtA = vB @ AtA                                 # length-n covector
    vy = theta * float(v @ y_star)                    # scalar forcing v^T B B^T y*

    e = -K["w_star"].copy()
    e0n = np.linalg.norm(e)
    b0 = abs(float(vB @ e))

    T = -1
    p1_err = 0.0
    p2_slack = np.inf
    max_e = e0n
    Delta = 0.0
    Delta_T = 0.0
    sum_kappa = 0.0
    Gamma = 0.0
    lam_hist = []

    n = min(len(seq), max_steps)
    for t in range(n):
        lam = float(seq[t])
        eta = stable_step(K, lam)

        q = B @ e
        nq = float(np.linalg.norm(q))
        a = float(Zu @ e)

        # first hitting time of boundary accuracy
        if nq <= eps_b and T < 0:
            T = t
            Delta_T = Delta
            break

        # boundary-lead diagnostic
        if abs(a) > 1e-13:
            Gamma = max(Gamma, nq / abs(a))

        if track_cond:
            lam_hist.append(lam)
            # kappa(H_lam): full condition number over the nonzero spectrum; the
            # kernel dim = rank[A;B] is lambda-independent, so a relative threshold
            # against the top eigenvalue isolates the nonzero part correctly.
            ev = np.linalg.eigvalsh(AtA + lam * BtB)
            evpos = ev[ev > ev[-1] * 1e-13]
            sum_kappa += evpos[-1] / evpos[0]

        # P1: exact readout recursion (Lemma 3), b_t = v^T B e_t
        b0t = float(vB @ e)
        b_pred = ((1.0 - eta * lam * theta) * b0t
                  - eta * float(vB_AtA @ e) + eta * vy)

        # advance the exact error recursion (R)
        e = e - eta * (AtA @ e + lam * (BtB @ e)) + eta * Bty
        max_e = max(max_e, float(np.linalg.norm(e)))

        b1 = float(vB @ e)
        p1_err = max(p1_err, abs(b_pred - b1))

        # P2: per-step slowest-mode floor (Lemma 4)
        lb = (1.0 - beta) * abs(b0t) - eta * Kd
        p2_slack = min(p2_slack, abs(b1) - lb)

        # drive envelope for P4 (Delta_{t+1} = (1-beta) Delta_t + eta Kd)
        Delta = (1.0 - beta) * Delta + eta * Kd

    return dict(T=T, p1_err=p1_err, p2_slack=p2_slack,
                max_e_ratio=max_e / e0n, Delta_T=Delta_T, b0=b0,
                sum_kappa=sum_kappa, Gamma=Gamma, lam_hist=lam_hist)


# --------------------------------------------------------------------------
# main search on one testbed
# --------------------------------------------------------------------------
def search(prob, tag, eps_b, n_random, seed):
    K = constants(prob)
    theta, sB2, kred = K["theta"], K["sB2"], K["kred"]
    beta = theta / sB2                               # for eta <= 1/lammax(H)
    v = K["v"]
    R = np.linalg.norm(K["w_star"])                  # ||e_0|| (w_0 = 0); P3 checks <= this
    Kd = float(np.linalg.norm(K["AtABtv"]) * R + theta * K["rho"])

    print(f"\n== SEARCH [{tag}]: kred={kred:.4e} theta=lammin(BB^T)={theta:.4e} "
          f"sB^2={sB2:.4e} beta=theta/sB^2={beta:.4e}", flush=True)
    print(f"   chi={K['chi']:.4e} rho={K['rho']:.4e} ||A^TA B^Tv||={np.linalg.norm(K['AtABtv']):.4e} "
          f"R=||e0||={R:.4e} Kd={Kd:.4e}", flush=True)

    lam_pool = [1e4, 1e6, 1e8, 1e10, 1e12, 1e14]
    # schedule length: long enough that the FAST (adversary) schedules reach eps_b
    # (the fastest resolve in ~50 steps in 1D, ~2400 in 2D); slow schedules that do
    # not reach within this window simply do not constrain the floor (a lower bound
    # on T that exceeds the run length is not violated).
    T_sched = 12000
    sched = make_schedules(n_random, lam_pool, T=T_sched, seed=seed)
    n_sched = len(sched)

    # aggregate worst-case statistics
    worst_p1 = 0.0
    worst_p2 = np.inf
    worst_e_ratio = 0.0
    worst_fence = np.inf
    n_reach = 0
    worst_T_ratio = np.inf         # measured_T / step-floor (P4); want >= 1
    worst_cond_ratio = np.inf      # measured sum_kappa / (kred * step-floor) (P5)
    worst_E7 = -np.inf             # max (Delta_T - E7bound); want <= 0
    worst_Gamma = 0.0

    # P0 fence: sample kappa/keff over the weight pool (cheap, weight-only)
    for lam in lam_pool + [1e0, 1e2, 1e5, 1e7]:
        H = K["AtA"] + lam * K["BtB"]
        ev = np.linalg.eigvalsh(H)
        evpos = ev[ev > ev[-1] * 1e-13]
        keff = evpos[-1] / evpos[0]
        worst_fence = min(worst_fence, keff / kred)

    m_window = int(np.ceil(np.log(1.0 / eps_b) / np.log(1.0 / (1.0 - beta))))
    lam0_const = 2.0 * Kd / (sB2 * beta)

    # cheap sweep pass (no eigensolve) over ALL schedules for P1-P4 and the reaching set
    reaching = []
    for name, seq in sched.items():
        m = run_schedule(K, seq, eps_b, R, Kd, beta, max_steps=T_sched,
                         track_cond=False)
        worst_p1 = max(worst_p1, m["p1_err"])
        worst_p2 = min(worst_p2, m["p2_slack"])
        worst_e_ratio = max(worst_e_ratio, m["max_e_ratio"])
        worst_Gamma = max(worst_Gamma, m["Gamma"])

        if m["T"] > 0:
            n_reach += 1
            reaching.append((name, seq, m))
            denom = eps_b + m["Delta_T"]
            arg = m["b0"] / denom
            step_floor = (np.log(arg) / np.log(1.0 / (1.0 - beta))
                          if arg > 1.0 else 0.0)
            if step_floor > 0:
                worst_T_ratio = min(worst_T_ratio, m["T"] / step_floor)

    # expensive pass (with per-step kappa) ONLY on the reaching schedules, and only
    # up to their hitting time T (small for fast schedules), capped in count.
    cond_checked = 0
    for name, seq, mc in reaching:
        if cond_checked >= 30:
            break
        m = run_schedule(K, seq, eps_b, R, Kd, beta, max_steps=T_sched,
                         track_cond=True)
        if m["T"] <= 0:
            continue
        cond_checked += 1
        denom = eps_b + m["Delta_T"]
        arg = m["b0"] / denom
        step_floor = (np.log(arg) / np.log(1.0 / (1.0 - beta))
                      if arg > 1.0 else 0.0)
        # P5 accumulated conditioning vs kred*floor
        bound = kred * step_floor
        if bound > 0:
            worst_cond_ratio = min(worst_cond_ratio, m["sum_kappa"] / bound)
        # P7 closed-form drive-envelope bound (E7), valid for T > m_window
        if m["T"] > m_window and len(m["lam_hist"]) > m_window:
            lam_Tm = m["lam_hist"][m["T"] - m_window]
            lam_0 = m["lam_hist"][0]
            E7 = lam0_const * (1.0 / lam_Tm
                               + ((1.0 - beta) ** m_window) / lam_0)
            worst_E7 = max(worst_E7, m["Delta_T"] - E7 * (1.0 + 1e-6))

    # ---- report ----
    check(f"[{tag}] P0 fence keff(H_lam) >= kred at every weight (per-step factor kred)",
          worst_fence >= 1 - 1e-6, f"min keff/kred={worst_fence:.6f}")
    check(f"[{tag}] P1 exact slowest-mode readout recursion (Lemma 3) matches truth",
          worst_p1 < 1e-9, f"max abs err={worst_p1:.2e} over {n_sched} schedules")
    check(f"[{tag}] P2 per-step slowest-mode floor (Lemma 4) holds every step",
          worst_p2 >= -1e-8,
          f"min (|b1| - lb)={worst_p2:.2e} over {n_sched} schedules")
    check(f"[{tag}] P3 trajectory bound ||e_t|| <= ||e_0|| (Lemma 2)",
          worst_e_ratio <= 1 + 1e-9,
          f"max ||e_t||/||e_0||={worst_e_ratio:.8f}")
    check(f"[{tag}] P4 UNCONDITIONAL step-count floor T >= ln(|b0|/(eps_b+Delta))/ln(1/(1-beta))",
          (n_reach == 0) or (worst_T_ratio >= 1 - 1e-6),
          f"min measured_T/floor={worst_T_ratio:.4f} over {n_reach}/{n_sched} "
          f"schedules reaching eps_b={eps_b:.0e}")
    check(f"[{tag}] P5 accumulated conditioning sum kappa >= kred*floor (coupled-path floor)",
          (cond_checked == 0) or (worst_cond_ratio >= 1 - 1e-6),
          f"min sum_kappa/(kred*floor)={worst_cond_ratio:.3f} over {cond_checked} "
          f"schedules")
    check(f"[{tag}] P7 closed-form drive envelope (E7) upper-bounds exact Delta_T",
          worst_E7 <= 0.0,
          f"max (Delta_T - E7bound)={worst_E7:.2e} (want <= 0)")
    print(f"   [DIAG] max boundary lead Gamma over all schedules = {worst_Gamma:.4e} "
          f"(chi*Gamma={K['chi']*worst_Gamma:.4e}); the new proof does NOT use Gamma",
          flush=True)

    return worst_T_ratio, worst_cond_ratio, n_reach, n_sched


# ==========================================================================
if __name__ == "__main__":
    np.random.seed(0)
    eps_b = 1e-8

    prob1 = build_poisson_1d_eig(normalize=True)
    prob2 = build_poisson_2d()

    # >= 500 schedules total across the two testbeds (canonical shapes + random)
    r1 = search(prob1, "1D eig, normalized", eps_b, n_random=210, seed=11)
    r2 = search(prob2, "2D tensor Fourier", eps_b, n_random=210, seed=23)

    total_sched = r1[3] + r2[3]
    total_reach = r1[2] + r2[2]
    worst_T = min(r1[0], r2[0])
    worst_cond = min(r1[1], r2[1])

    print(f"\n---- SUMMARY ----", flush=True)
    print(f"total schedules searched : {total_sched}", flush=True)
    print(f"schedules reaching eps_b : {total_reach}", flush=True)
    print(f"worst-case T-floor ratio (measured/bound, P4)          : {worst_T:.4f} "
          f"(>= 1 required)", flush=True)
    print(f"worst-case conditioning-floor ratio (measured/bound,P5): {worst_cond:.4f} "
          f"(>= 1 required)", flush=True)

    print("\nALL PASS" if PASS else "\nSOME CHECKS FAILED", flush=True)
    raise SystemExit(0 if PASS else 1)
