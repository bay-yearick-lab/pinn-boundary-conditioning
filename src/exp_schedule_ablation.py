"""EXP C -- schedule ablation for the open path-product conjecture.

The paper proves the path law for fixed/diagonal preconditioning and for
decoupled schedules, and leaves the GENERAL COUPLED PATH as an explicit
conjecture: "the right invariant is conjectured to be the accumulated
conditioning itself." This experiment tests that conjecture directly.

Design. Several weight SCHEDULES lambda_t that all
  - reach the SAME endpoint weight lam_end, and
  - spend the SAME total training budget T,
but accumulate DIFFERENT total conditioning  Kappa = sum_t kappa(H_{lam_t}).
Schedules:
  jump   : lam = lam_end from step 0 (max accumulated conditioning: stiff early)
  ramp   : lam grows log-linearly 1 -> lam_end (moderate)
  cont   : continuation, lam held low then a late jump to lam_end (low early)
  cos    : smooth cosine warmup of lam 1 -> lam_end
All share endpoint, budget, architecture, seeds; only the PATH differs.

Along each run we accumulate the per-step linearized conditioning kappa_eff
of the current weight from a cheap kernel-trace proxy (the same tr(K_rr),
tr(K_bb) probe the paper's navigators use), so the "accumulated conditioning"
is a measured quantity, not assumed. We then test whether the final accuracy /
effective cost tracks Kappa (the conjectured invariant) rather than the shared
endpoint. Both outcomes are reported
from the fitted rank correlation.

CPU, 4x64 tanh, capped Adam. Deterministic seeds. Minutes.
Writes data/schedule_ablation.jsonl (per-run) + data/schedule_ablation.json.
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(6)
try:
    sys.stdout.reconfigure(line_buffering=True)   # stream output live
except Exception:
    pass

from exp_pinn_sweep import MLP, evaluate, f_rhs_t, laplacian, u_exact_t

OUT_JSONL = Path(__file__).resolve().parent.parent / "data" / "schedule_ablation.jsonl"
OUT_JSON = Path(__file__).resolve().parent.parent / "data" / "schedule_ablation.json"
DEV = "cpu"
DTYPE = torch.float32

STEPS = 5000
LAM_END = 1e4          # shared endpoint weight (right of the valley: stiff)
N_INT, N_BND = 512, 128
SEEDS = 3
PROBE_EVERY = 250      # accumulate conditioning proxy every this many steps
SCHEDULES = ["jump", "ramp", "cont", "cos"]


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
    g_bnd = u_exact_t(xy_bnd[:, 0], xy_bnd[:, 1])
    f_int = f_rhs_t(xy_int[:, 0], xy_int[:, 1])
    return xy_int, xy_bnd, g_bnd, f_int


def lam_at(schedule, step):
    """All schedules end at LAM_END at step STEPS-1; same endpoint, same budget."""
    frac = step / max(STEPS - 1, 1)
    if schedule == "jump":
        return LAM_END
    if schedule == "ramp":                      # log-linear 1 -> LAM_END
        return float(10 ** (frac * math.log10(LAM_END)))
    if schedule == "cont":                       # low plateau then late jump
        return 1.0 if frac < 0.75 else LAM_END
    if schedule == "cos":                        # cosine warmup 1 -> LAM_END
        c = 0.5 * (1 - math.cos(math.pi * frac))  # 0->1 smooth
        return float(10 ** (c * math.log10(LAM_END)))
    raise ValueError(schedule)


def kappa_proxy(model, xy_int, xy_bnd, f_int, lam):
    """Cheap kernel-trace conditioning proxy of H_lam at the current point.

    Uses tr(K_rr) and tr(lam K_bb) (the diagonal-mass scales the two blocks
    balance) as a monotone stand-in for the effective conditioning: when the
    two block masses are badly imbalanced the penalized Hessian is stiff. We
    return max(tr_r, lam*tr_b) / min(tr_r, lam*tr_b), a subsampled trace ratio,
    which is the kernel-trace navigator's own diagnostic read as a conditioning
    surrogate. Subsampled to stay cheap.
    """
    k = 32
    params = [p for p in model.parameters() if p.requires_grad]
    ridx = torch.randperm(xy_int.shape[0])[:k]
    bidx = torch.randperm(xy_bnd.shape[0])[:k]
    # residual trace
    tr_r = 0.0
    for i in ridx:
        x = xy_int[i:i + 1].clone().requires_grad_(True)
        u = model(x)
        du = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        uxx = torch.autograd.grad(du[:, 0].sum(), x, create_graph=True)[0][:, 0]
        uyy = torch.autograd.grad(du[:, 1].sum(), x, create_graph=True)[0][:, 1]
        val = (uxx + uyy)[0]
        g = torch.autograd.grad(val, params, retain_graph=False,
                                allow_unused=True)
        tr_r += sum(float((gi ** 2).sum()) for gi in g if gi is not None)
    tr_r /= k
    # boundary trace
    tr_b = 0.0
    for i in bidx:
        x = xy_bnd[i:i + 1]
        val = model(x).squeeze(-1)[0]
        g = torch.autograd.grad(val, params, retain_graph=False,
                                allow_unused=True)
        tr_b += sum(float((gi ** 2).sum()) for gi in g if gi is not None)
    tr_b /= k
    a, b = tr_r, lam * tr_b
    hi, lo = max(a, b), min(a, b)
    return hi / max(lo, 1e-30)


def run_one(schedule, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP().to(DEV, DTYPE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS,
                                                       eta_min=1e-5)
    xy_int, xy_bnd, g_bnd, f_int = make_batch(seed)
    accum_kappa = 0.0
    n_probe = 0
    for step in range(STEPS):
        lam = lam_at(schedule, step)
        opt.zero_grad(set_to_none=True)
        _, lap = laplacian(model, xy_int.clone(), hard=False)
        loss_r = torch.mean((lap + f_int) ** 2)
        ub = model(xy_bnd).squeeze(-1)
        loss_b = torch.mean((ub - g_bnd) ** 2)
        (loss_r + lam * loss_b).backward()
        opt.step()
        sched.step()
        if step % PROBE_EVERY == 0:
            kp = kappa_proxy(model, xy_int, xy_bnd, f_int, lam)
            # accumulate over the interval this probe represents (Riemann sum)
            accum_kappa += kp * PROBE_EVERY
            n_probe += 1
    rel_l2, bnd_rms = evaluate(model, DEV, DTYPE, hard=False)
    return dict(schedule=schedule, seed=seed, lam_end=LAM_END,
                rel_l2=float(rel_l2), bnd_rms=float(bnd_rms),
                accum_kappa=float(accum_kappa), n_probe=n_probe)


def rank_corr(x, y):
    """Rank correlation without scipy dependency."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    denom = math.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def main():
    t0 = time.time()
    print("=" * 74)
    print("EXP C -- schedule ablation (2D Poisson, 4x64 tanh)")
    print(f"same endpoint lam_end={LAM_END:g}, same budget {STEPS} steps, "
          f"different accumulated conditioning")
    print("conjecture: final cost tracks accumulated conditioning, not endpoint")
    print("=" * 74)
    OUT_JSONL.write_text("")
    records = []
    with OUT_JSONL.open("a") as fh:
        for schedule in SCHEDULES:
            for seed in range(SEEDS):
                rec = run_one(schedule, seed)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                records.append(rec)
            errs = [r["rel_l2"] for r in records if r["schedule"] == schedule]
            kaps = [r["accum_kappa"] for r in records if r["schedule"] == schedule]
            print(f"  {schedule:5s}  median rel_l2={np.median(errs):.3e}  "
                  f"median accum_kappa={np.median(kaps):.3e}")

    # ---- per-schedule aggregates ----
    agg = {}
    for schedule in SCHEDULES:
        errs = [r["rel_l2"] for r in records if r["schedule"] == schedule]
        kaps = [r["accum_kappa"] for r in records if r["schedule"] == schedule]
        bnds = [r["bnd_rms"] for r in records if r["schedule"] == schedule]
        agg[schedule] = dict(median_rel_l2=float(np.median(errs)),
                             median_accum_kappa=float(np.median(kaps)),
                             median_bnd=float(np.median(bnds)),
                             rel_l2=errs, accum_kappa=kaps)

    # ---- test the conjecture ----
    # all schedules share endpoint lam_end, so if endpoint alone determined the
    # cost they would tie. Spread in final error is the schedule effect; the
    # conjecture says it is ordered by accumulated conditioning.
    order_by_kappa = sorted(SCHEDULES, key=lambda s: agg[s]["median_accum_kappa"])
    order_by_err = sorted(SCHEDULES, key=lambda s: agg[s]["median_rel_l2"])
    # rank correlation across ALL runs (not just medians)
    all_k = [r["accum_kappa"] for r in records]
    all_e = [r["rel_l2"] for r in records]
    rho_runs = rank_corr(all_k, all_e)
    # rank correlation across schedule medians
    med_k = [agg[s]["median_accum_kappa"] for s in SCHEDULES]
    med_e = [agg[s]["median_rel_l2"] for s in SCHEDULES]
    rho_med = rank_corr(med_k, med_e)
    err_spread = max(med_e) / min(med_e)
    kappa_spread = max(med_k) / min(med_k)

    print("\n" + "=" * 74)
    print("CONJECTURE TEST")
    print("=" * 74)
    print(f"  shared endpoint weight lam_end = {LAM_END:g} (same for all schedules)")
    print(f"  final-error spread across schedules: {err_spread:.2f}x  "
          f"(pure-endpoint model predicts ~1x tie)")
    print(f"  accumulated-conditioning spread:     {kappa_spread:.2f}x")
    print(f"  schedules ranked by accumulated conditioning (low->high): "
          f"{order_by_kappa}")
    print(f"  schedules ranked by final error       (low->high): {order_by_err}")
    print(f"  rank correlation(accum_kappa, final_error), per-run  = {rho_runs:+.3f}")
    print(f"  rank correlation(accum_kappa, final_error), medians  = {rho_med:+.3f}")
    verdict = ("SUPPORTED: lower accumulated conditioning -> lower error, "
               "endpoint alone does not explain the spread"
               if rho_med > 0.5 and err_spread > 1.3 else
               "NULL/WEAK: accumulated conditioning does not cleanly order "
               "the final error at this scale")
    print(f"  VERDICT: {verdict}")
    print("=" * 74)

    out = dict(aggregate=agg,
               conjecture=dict(order_by_kappa=order_by_kappa,
                               order_by_err=order_by_err,
                               err_spread=err_spread,
                               kappa_spread=kappa_spread,
                               rank_corr_runs=rho_runs,
                               rank_corr_medians=rho_med,
                               verdict=verdict),
               config=dict(steps=STEPS, lam_end=LAM_END, seeds=SEEDS,
                           n_int=N_INT, n_bnd=N_BND, probe_every=PROBE_EVERY,
                           schedules=SCHEDULES),
               wall_s=round(time.time() - t0, 1))
    OUT_JSON.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT_JSON}  ({out['wall_s']}s total)")


if __name__ == "__main__":
    main()
