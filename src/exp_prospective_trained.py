"""EXP A -- PROSPECTIVE trained-network diagnostic.

The genuinely prospective protocol (order is load-bearing and enforced):

  1. Warm up a 4x64 tanh PINN on the 2D Poisson problem for a FIXED short
     budget at a neutral weight (lam_warm), then FREEZE. Nothing about the
     confirming sweep has run yet.
  2. MEASURE the tangent-kernel constants at the warmup point, kernel-side,
     exactly in the style of exp_ntk_constants.py: the shadow price rho, the
     reduced curvature mu, kappa_red, the boundary Schur curvature nu, and the
     boundary operator scale sigma_B. From these form the linearized effective
     conditioning kappa_eff(lam) across the weight grid.
  3. COMMIT (print + write to disk) the pre-registered predictions BEFORE any
     confirming run:
        - the conditioning-valley weight = argmin_lam kappa_eff(lam);
        - the predicted best-accuracy region (the flat valley floor of
          kappa_eff, weights within 10% of the minimum);
        - the predicted product limit rho * sigma_B^2 / mu;
        - the predicted failure mode (which branch, starved vs drag, and whether
          the hard twin should win on statics).
  4. ONLY THEN run the confirming sweep: several lam, few seeds, each a full
     (short) Adam train, plus a hard-constrained twin at the same budget.
  5. CONFIRM whether the observed best-accuracy region and hard-twin advantage
     match the pre-committed prediction.

Everything is CPU, small, capped. Deterministic seeds. A few minutes.

Writes data/prospective_trained.json (prediction block first, then the
confirming sweep, then the scored comparison).
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(6)  # shared CPU with sibling agents

from exp_pinn_sweep import (MLP, transfinite_lifting, evaluate, f_rhs_t, laplacian,
                            u_exact_t)

OUT = Path(__file__).resolve().parent.parent / "data" / "prospective_trained.json"
DEV = "cpu"
DTYPE = torch.float32

# ----- fixed protocol constants (all pre-registered) -----
WARM_STEPS = 3000      # fixed warmup budget for the measurement point
LAM_WARM = 100.0       # neutral warmup weight
SWEEP_STEPS = 6000     # confirming-sweep budget per run
SWEEP_LAMS = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
SWEEP_SEEDS = 3
N_INT, N_BND = 1024, 128


def make_batch(seed):
    gen = torch.Generator(device="cpu").manual_seed(seed + 1)
    xy_int = torch.rand((N_INT, 2), generator=gen).to(DEV, DTYPE)
    tb = torch.rand((N_BND,), generator=gen).to(DEV, DTYPE)
    edge = (torch.arange(N_BND) % 4).to(DEV)
    zx, ox = torch.zeros_like(tb), torch.ones_like(tb)
    bx = torch.where(edge == 0, tb, torch.where(edge == 1, tb,
                     torch.where(edge == 2, zx, ox)))
    by = torch.where(edge == 0, zx, torch.where(edge == 1, ox,
                     torch.where(edge == 2, tb, tb)))
    xy_bnd = torch.stack([bx, by], dim=1)
    return xy_int, xy_bnd


def train(mode, lam, seed, steps, model=None):
    """One (short) Adam train, cosine-annealed, exactly as the paper's sweep."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if model is None:
        model = MLP().to(DEV, DTYPE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps,
                                                       eta_min=1e-5)
    xy_int, xy_bnd = make_batch(seed)
    g_bnd = u_exact_t(xy_bnd[:, 0], xy_bnd[:, 1])
    f_int = f_rhs_t(xy_int[:, 0], xy_int[:, 1])
    hard = mode == "hard"
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        u_i, lap = laplacian(model, xy_int.clone(), hard=hard)
        res = lap + f_int
        loss = torch.mean(res ** 2)
        if not hard:
            u_b = model(xy_bnd).squeeze(-1)
            loss = loss + lam * torch.mean((u_b - g_bnd) ** 2)
        loss.backward()
        opt.step()
        sched.step()
    return model


# ---------- tangent-kernel constant measurement (kernel-side) ----------
def residual_row(model, xy_one):
    x = xy_one.clone().requires_grad_(True)
    u = model(x)
    du = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    uxx = torch.autograd.grad(du[:, 0].sum(), x, create_graph=True)[0][:, 0]
    uyy = torch.autograd.grad(du[:, 1].sum(), x, create_graph=True)[0][:, 1]
    return (uxx + uyy + f_rhs_t(x[:, 0], x[:, 1]))[0]


def boundary_row(model, xy_one):
    return (model(xy_one).squeeze(-1) - u_exact_t(xy_one[:, 0], xy_one[:, 1]))[0]


def residual_row_hard(model, xy_one):
    x = xy_one.clone().requires_grad_(True)
    xx, yy = x[:, :1], x[:, 1:]
    u = transfinite_lifting(xx, yy) + xx * (1 - xx) * yy * (1 - yy) * model(x)
    du = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    uxx = torch.autograd.grad(du[:, 0].sum(), x, create_graph=True)[0][:, 0]
    uyy = torch.autograd.grad(du[:, 1].sum(), x, create_graph=True)[0][:, 1]
    return (uxx + uyy + f_rhs_t(x[:, 0], x[:, 1]))[0]


def jacobian_rows(model, xy, fn):
    params = [p for p in model.parameters() if p.requires_grad]
    rows, vals = [], []
    for i in range(xy.shape[0]):
        v = fn(model, xy[i:i + 1])
        g = torch.autograd.grad(v, params, retain_graph=False, allow_unused=True)
        g = [gi if gi is not None else torch.zeros_like(p)
             for gi, p in zip(g, params)]
        rows.append(torch.cat([gi.reshape(-1) for gi in g]).detach())
        vals.append(float(v.item()))
    return torch.stack(rows), np.array(vals)


def kappa_eff_from_gram(G, tol=1e-9):
    ev = np.linalg.eigvalsh(G)
    nz = ev[ev > ev[-1] * tol]
    return ev[-1] / nz[0]


def measure_constants(model):
    """Kernel-side tangent constants at the warmup point (see exp_ntk_constants)."""
    xy_int, xy_bnd = make_batch(seed=777)  # measurement points, held out
    m, b = xy_int.shape[0], xy_bnd.shape[0]
    Jr, r0 = jacobian_rows(model, xy_int, residual_row)
    Jb, b0 = jacobian_rows(model, xy_bnd, boundary_row)
    Jr = (Jr / math.sqrt(m)).cpu().numpy().astype(np.float64)
    Jb = (Jb / math.sqrt(b)).cpu().numpy().astype(np.float64)
    r0 = r0 / math.sqrt(m)
    b0 = b0 / math.sqrt(b)

    Krr, Kbb, Krb = Jr @ Jr.T, Jb @ Jb.T, Jr @ Jb.T

    lams = np.logspace(-1, 6, 29)
    kaps = []
    for lam in lams:
        s = math.sqrt(lam)
        G = np.block([[Krr, s * Krb], [s * Krb.T, lam * Kbb]])
        kaps.append(float(kappa_eff_from_gram(G)))

    Kbb_inv = np.linalg.pinv(Kbb, rcond=1e-10)
    K_red = Krr - Krb @ Kbb_inv @ Krb.T
    kap_red = float(kappa_eff_from_gram(K_red))
    evr = np.linalg.eigvalsh(K_red)
    evr = evr[evr > evr[-1] * 1e-10]
    mu = float(evr[0])           # reduced curvature (smallest nonzero)
    # boundary Schur curvature nu
    AYm = Krb @ Kbb_inv
    evr2, Ur = np.linalg.eigh(K_red)
    nzmask = evr2 > evr2[-1] * 1e-10
    Pr = Ur[:, nzmask] @ Ur[:, nzmask].T
    Nmat = AYm.T @ (np.eye(m) - Pr) @ AYm
    nu = float(np.linalg.norm(Nmat, 2))
    sigmaB = float(np.linalg.norm(Jb, 2))
    # shadow price rho of the linearized constrained problem at the point
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


def measure_hard_clock(model_hard):
    xy_int, _ = make_batch(seed=777)
    m = xy_int.shape[0]
    Jr_h, _ = jacobian_rows(model_hard, xy_int, residual_row_hard)
    Jr_h = (Jr_h / math.sqrt(m)).cpu().numpy().astype(np.float64)
    return float(kappa_eff_from_gram(Jr_h @ Jr_h.T))


def main():
    t0 = time.time()
    print("=" * 74)
    print("EXP A -- PROSPECTIVE trained-network diagnostic (2D Poisson, 4x64 tanh)")
    print("=" * 74)

    # ---- step 1: fixed warmup, then freeze ----
    print(f"[1] warmup: {WARM_STEPS} Adam steps at lam={LAM_WARM:g}, seed 0 ...")
    warm = train("soft", LAM_WARM, 0, WARM_STEPS)
    wr, wb = evaluate(warm, DEV, DTYPE, hard=False)
    print(f"    warmup point rel_l2={wr:.3e}  bnd_rms={wb:.3e}")

    # brief hard warmup (same budget) so the hard clock is measured at a
    # comparable operating point, not at initialization
    warm_hard = train("hard", None, 0, WARM_STEPS)

    # ---- step 2: measure constants at the warmup point ----
    print("[2] measuring tangent-kernel constants at the warmup point ...")
    C = measure_constants(warm)
    kap_hard = measure_hard_clock(warm_hard)
    lams = np.array(C["lams"])
    kaps = np.array(C["kappas"])
    rho, mu, nu, kap_red, sigmaB = (C["rho"], C["mu"], C["nu"],
                                    C["kappa_red"], C["sigmaB"])

    # ---- step 3: COMMIT predictions (printed BEFORE the confirming sweep) ----
    lam_valley = float(lams[int(np.argmin(kaps))])
    kap_min = float(kaps.min())
    flat_mask = kaps <= 1.10 * kap_min          # within 10% of the floor
    valley_lo = float(lams[flat_mask].min())
    valley_hi = float(lams[flat_mask].max())
    product_pred = rho * sigmaB ** 2 / mu
    hard_should_win = kap_red < kap_min          # reduced clock beats best soft
    prediction = dict(
        lam_valley=lam_valley,
        valley_region=[valley_lo, valley_hi],
        kappa_eff_min=kap_min,
        kappa_eff_flat_ratio=float(kaps.max() / kap_min),
        product_limit=product_pred,
        kappa_red=kap_red,
        kappa_hard_tangent=kap_hard,
        hard_should_win_on_statics=bool(hard_should_win),
        rho=rho, mu=mu, nu=nu, sigmaB=sigmaB,
    )
    print("\n" + "#" * 74)
    print("# PRE-REGISTERED PREDICTION (committed BEFORE any confirming run)")
    print("#" * 74)
    print(f"  constants: rho={rho:.4e}  mu={mu:.4e}  nu={nu:.4e}  "
          f"sigma_B={sigmaB:.4e}  kappa_red={kap_red:.4e}")
    print(f"  kappa_eff(lam) min = {kap_min:.4e} at lam = {lam_valley:g}")
    print(f"  kappa_eff flat to {prediction['kappa_eff_flat_ratio']:.2f}x over "
          f"[{lams.min():g}, {lams.max():g}]")
    print(f"  --> PREDICTED conditioning valley weight: lam = {lam_valley:g}")
    print(f"  --> PREDICTED best-accuracy region (flat floor): "
          f"[{valley_lo:g}, {valley_hi:g}]")
    print(f"  --> PREDICTED product limit rho*sigma_B^2/mu = {product_pred:.4e}")
    print(f"  --> PREDICTED hard twin beats soft on statics: {hard_should_win} "
          f"(kappa_red={kap_red:.3e} vs best soft kappa_eff={kap_min:.3e})")
    fail_mode = ("flat wide valley: mis-weighting cheap inside "
                 f"[{valley_lo:g},{valley_hi:g}], drag/starvation only outside; "
                 "hard twin wins on statics" if hard_should_win else
                 "flat wide valley; hard twin has no static edge here")
    print(f"  --> PREDICTED failure mode: {fail_mode}")
    print("#" * 74 + "\n")

    # persist the prediction block immediately (before the sweep is even run)
    OUT.write_text(json.dumps(dict(prediction=prediction, constants=C,
                                   status="prediction_committed"), indent=1))

    # ---- step 4: confirming sweep (soft grid + hard twin) ----
    print(f"[4] confirming sweep: lams={SWEEP_LAMS}, {SWEEP_SEEDS} seeds, "
          f"{SWEEP_STEPS} steps each ...")
    sweep = {}
    for lam in SWEEP_LAMS:
        errs, bnds = [], []
        for seed in range(SWEEP_SEEDS):
            m = train("soft", lam, seed, SWEEP_STEPS)
            r, bnd = evaluate(m, DEV, DTYPE, hard=False)
            errs.append(r)
            bnds.append(bnd)
        sweep[f"{lam:g}"] = dict(rel_l2=errs, bnd_rms=bnds,
                                 median_rel_l2=float(np.median(errs)),
                                 median_bnd=float(np.median(bnds)))
        print(f"    soft lam={lam:8g}  median rel_l2={np.median(errs):.3e}  "
              f"median bnd={np.median(bnds):.3e}")

    hard_errs, hard_bnds = [], []
    for seed in range(SWEEP_SEEDS + 1):     # one extra seed for the twin
        m = train("hard", None, seed, SWEEP_STEPS)
        r, bnd = evaluate(m, DEV, DTYPE, hard=True)
        hard_errs.append(r)
        hard_bnds.append(bnd)
    hard_med = float(np.median(hard_errs))
    print(f"    hard twin      median rel_l2={hard_med:.3e}  "
          f"median bnd={np.median(hard_bnds):.3e}  (seeds {SWEEP_SEEDS + 1})")

    # ---- step 5: score prediction vs observation ----
    lam_keys = list(sweep.keys())
    meds = np.array([sweep[k]["median_rel_l2"] for k in lam_keys])
    lam_best_obs = float(lam_keys[int(np.argmin(meds))].replace("g", "") or 0) \
        if False else float(SWEEP_LAMS[int(np.argmin(meds))])
    best_obs_med = float(meds.min())
    # observed best-accuracy region: weights within 2x of the best median
    obs_region_mask = meds <= 2.0 * best_obs_med
    obs_lams = [SWEEP_LAMS[i] for i in range(len(SWEEP_LAMS)) if obs_region_mask[i]]
    valley_hit = bool(valley_lo / 1.001 <= lam_best_obs <= valley_hi * 1.001) or \
        bool(lam_best_obs in obs_lams and
             any(valley_lo <= L <= valley_hi for L in obs_lams))
    # was the predicted valley region non-trivially overlapping observed region?
    region_overlap = [L for L in obs_lams if valley_lo <= L <= valley_hi]
    hard_advantage = best_obs_med / hard_med
    hard_win_obs = hard_med < best_obs_med
    hard_pred_correct = (hard_should_win == hard_win_obs)

    confirmation = dict(
        lam_best_observed=lam_best_obs,
        best_soft_median=best_obs_med,
        observed_best_region=obs_lams,
        predicted_valley_region=[valley_lo, valley_hi],
        predicted_valley_in_observed_region=bool(region_overlap),
        hard_median=hard_med,
        hard_advantage_x=hard_advantage,
        hard_won_observed=hard_win_obs,
        hard_prediction_correct=hard_pred_correct,
    )
    print("\n" + "=" * 74)
    print("SCORING (prediction vs observation)")
    print("=" * 74)
    print(f"  predicted valley region [{valley_lo:g}, {valley_hi:g}]  "
          f"(argmin kappa_eff at {lam_valley:g})")
    print(f"  observed best weight    {lam_best_obs:g}  "
          f"(best-region weights {obs_lams})")
    print(f"  predicted region overlaps observed best region: "
          f"{bool(region_overlap)}  (overlap={region_overlap})")
    print(f"  hard twin advantage over best soft: {hard_advantage:.2f}x  "
          f"(hard={hard_med:.3e} vs soft={best_obs_med:.3e})")
    print(f"  predicted hard-wins={hard_should_win}, observed={hard_win_obs}  "
          f"--> prediction correct: {hard_pred_correct}")
    print("=" * 74)

    full = dict(status="confirmed", prediction=prediction, constants=C,
                sweep=sweep, hard=dict(rel_l2=hard_errs, bnd_rms=hard_bnds,
                                       median_rel_l2=hard_med),
                confirmation=confirmation,
                config=dict(warm_steps=WARM_STEPS, lam_warm=LAM_WARM,
                            sweep_steps=SWEEP_STEPS, sweep_lams=SWEEP_LAMS,
                            sweep_seeds=SWEEP_SEEDS, n_int=N_INT, n_bnd=N_BND),
                wall_s=round(time.time() - t0, 1))
    OUT.write_text(json.dumps(full, indent=1))
    print(f"\nwrote {OUT}  ({full['wall_s']}s total)")


if __name__ == "__main__":
    main()
