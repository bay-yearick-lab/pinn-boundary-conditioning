"""EXP B -- minibatch-noise ablation: the second, stochastic penalty cost.

The paper's product law is proved full-batch. This experiment prices the
minibatch collocation noise, whose boundary component scales with the weight.

Claim under test. When the boundary loss is estimated on a random subsample of
the boundary collocation set each step, the penalty term lam * mse_b carries a
sampling variance that does not vanish as the optimizer converges: it sets a
STOCHASTIC FLOOR on the steady-state boundary error (leak). The full-batch run
is the clean control (no subsampling variance). The prediction from a simple
SGD-noise argument: the stationary leak floor grows with the weight lam and
shrinks as the batch size B grows, so that

    leak_floor(lam, B)  ~  C * sqrt(lam / B)      (a second stochastic cost)

up to a full-batch/deterministic baseline it must exceed. We measure the
steady-state boundary error across a grid of (lam, batch size), subtract the
full-batch floor in quadrature, and fit the two scaling exponents. The
measured exponents are reported as fitted.

Fixed 2D Poisson, 4x64 tanh, capped Adam. CPU, deterministic seeds, minutes.
Writes data/minibatch_noise.jsonl (per-run) + data/minibatch_noise.json (fit).
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

OUT_JSONL = Path(__file__).resolve().parent.parent / "data" / "minibatch_noise.jsonl"
OUT_JSON = Path(__file__).resolve().parent.parent / "data" / "minibatch_noise.json"
DEV = "cpu"
DTYPE = torch.float32

STEPS = 4000            # capped budget (CPU double-backward is the cost)
MEASURE_TAIL = 1000     # average the leak over the last window (steady state)
N_INT = 512
N_BND_POOL = 256        # full boundary collocation pool
SEEDS = 2
LAMS = [1e1, 1e2, 1e3, 1e4]
BATCHES = [8, 32, 128, 256]   # 256 == full batch (deterministic control)


def make_pool(seed):
    gen = torch.Generator(device="cpu").manual_seed(seed + 1)
    xy_int = torch.rand((N_INT, 2), generator=gen).to(DEV, DTYPE)
    tb = torch.rand((N_BND_POOL,), generator=gen).to(DEV, DTYPE)
    edge = (torch.arange(N_BND_POOL) % 4).to(DEV)
    zx, ox = torch.zeros_like(tb), torch.ones_like(tb)
    bx = torch.where(edge == 0, tb, torch.where(edge == 1, tb,
                     torch.where(edge == 2, zx, ox)))
    by = torch.where(edge == 0, zx, torch.where(edge == 1, ox,
                     torch.where(edge == 2, tb, tb)))
    xy_bnd = torch.stack([bx, by], dim=1)
    g_bnd = u_exact_t(xy_bnd[:, 0], xy_bnd[:, 1])
    f_int = f_rhs_t(xy_int[:, 0], xy_int[:, 1])
    return xy_int, xy_bnd, g_bnd, f_int


def run_one(lam, batch, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP().to(DEV, DTYPE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS,
                                                       eta_min=1e-5)
    xy_int, xy_bnd, g_bnd, f_int = make_pool(seed)
    full = batch >= N_BND_POOL
    bgen = torch.Generator(device="cpu").manual_seed(seed + 12345)

    tail_leaks = []
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        _, lap = laplacian(model, xy_int.clone(), hard=False)
        loss_r = torch.mean((lap + f_int) ** 2)
        if full:
            ub = model(xy_bnd).squeeze(-1)
            loss_b = torch.mean((ub - g_bnd) ** 2)
        else:
            idx = torch.randperm(N_BND_POOL, generator=bgen)[:batch]
            ub = model(xy_bnd[idx]).squeeze(-1)
            loss_b = torch.mean((ub - g_bnd[idx]) ** 2)
        (loss_r + lam * loss_b).backward()
        opt.step()
        sched.step()
        if step >= STEPS - MEASURE_TAIL:
            with torch.no_grad():
                ub_full = model(xy_bnd).squeeze(-1)
                # true (full-pool) RMS boundary error = the leak
                leak = torch.sqrt(torch.mean((ub_full - g_bnd) ** 2)).item()
            tail_leaks.append(leak)
    _, bnd_full_eval = evaluate(model, DEV, DTYPE, hard=False)
    rel_l2, _ = evaluate(model, DEV, DTYPE, hard=False)
    return dict(lam=lam, batch=batch, seed=seed,
                leak_mean=float(np.mean(tail_leaks)),
                leak_std=float(np.std(tail_leaks)),
                leak_eval=float(bnd_full_eval),
                rel_l2=float(rel_l2))


def loglog_slope(x, y):
    x, y = np.log(np.asarray(x)), np.log(np.asarray(y))
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(coef[0]), float(r2)


def main():
    t0 = time.time()
    print("=" * 74)
    print("EXP B -- minibatch-noise ablation (2D Poisson, 4x64 tanh)")
    print("prediction: leak_floor ~ sqrt(lam / batch), full-batch = clean control")
    print("=" * 74)
    OUT_JSONL.write_text("")
    records = []
    with OUT_JSONL.open("a") as fh:
        for lam in LAMS:
            for batch in BATCHES:
                leaks = []
                for seed in range(SEEDS):
                    rec = run_one(lam, batch, seed)
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    records.append(rec)
                    leaks.append(rec["leak_mean"])
                tag = "FULL" if batch >= N_BND_POOL else f"B={batch}"
                print(f"  lam={lam:8g}  {tag:6s}  leak(steady) median="
                      f"{np.median(leaks):.4e}  seeds={[f'{x:.2e}' for x in leaks]}")

    # ---- aggregate: median steady-state leak per (lam, batch) ----
    def med(lam, batch):
        vals = [r["leak_mean"] for r in records
                if r["lam"] == lam and r["batch"] == batch]
        return float(np.median(vals))

    grid = {f"{lam:g}": {str(b): med(lam, b) for b in BATCHES} for lam in LAMS}

    # ---- full-batch floor (deterministic control) and excess (stochastic) leak ----
    # excess leak_floor = sqrt(max(leak^2 - leak_full^2, eps)): the part above
    # the clean full-batch baseline, i.e. the stochastic penalty cost.
    print("\n  stochastic excess leak (subtracting full-batch floor in quadrature):")
    excess = {}
    for lam in LAMS:
        leak_full = med(lam, BATCHES[-1])   # 512 = full
        excess[f"{lam:g}"] = {}
        for b in BATCHES[:-1]:              # stochastic batches only
            lk = med(lam, b)
            ex = math.sqrt(max(lk ** 2 - leak_full ** 2, 0.0))
            excess[f"{lam:g}"][str(b)] = ex
        row = "  ".join(f"B={b}:{excess[f'{lam:g}'][str(b)]:.3e}"
                        for b in BATCHES[:-1])
        print(f"    lam={lam:8g}  full_floor={leak_full:.3e}   {row}")

    # ---- scaling fits ----
    fits = {}
    # (a) leak_floor vs lam at each fixed batch (raw steady leak)
    print("\n  leak_floor vs lam  (log-log slope per batch):")
    fits["leak_vs_lam_per_batch"] = {}
    for b in BATCHES:
        ys = [med(lam, b) for lam in LAMS]
        s, r2 = loglog_slope(LAMS, ys)
        fits["leak_vs_lam_per_batch"][str(b)] = dict(slope=s, r2=r2)
        tag = "FULL" if b >= N_BND_POOL else f"B={b}"
        print(f"    {tag:6s}  slope={s:+.3f}  r2={r2:.3f}")

    # (b) excess (stochastic) leak vs lam at each stochastic batch
    print("\n  stochastic-excess leak vs lam  (log-log slope per batch):")
    fits["excess_vs_lam_per_batch"] = {}
    for b in BATCHES[:-1]:
        ys = [excess[f"{lam:g}"][str(b)] for lam in LAMS]
        ys = [max(v, 1e-12) for v in ys]
        s, r2 = loglog_slope(LAMS, ys)
        fits["excess_vs_lam_per_batch"][str(b)] = dict(slope=s, r2=r2)
        print(f"    B={b:4d}  slope={s:+.3f}  r2={r2:.3f}  "
              f"(predicted +0.5)")

    # (c) excess leak vs batch at each lam
    print("\n  stochastic-excess leak vs batch  (log-log slope per lam):")
    fits["excess_vs_batch_per_lam"] = {}
    sb = BATCHES[:-1]
    for lam in LAMS:
        ys = [max(excess[f"{lam:g}"][str(b)], 1e-12) for b in sb]
        s, r2 = loglog_slope(sb, ys)
        fits["excess_vs_batch_per_lam"][f"{lam:g}"] = dict(slope=s, r2=r2)
        print(f"    lam={lam:8g}  slope={s:+.3f}  r2={r2:.3f}  "
              f"(predicted -0.5)")

    # ---- pooled two-exponent fit: log(excess) = a*log(lam) + b*log(batch) + c ----
    X, Y = [], []
    for lam in LAMS:
        for b in BATCHES[:-1]:
            ex = excess[f"{lam:g}"][str(b)]
            if ex > 1e-12:
                X.append([math.log(lam), math.log(b), 1.0])
                Y.append(math.log(ex))
    X, Y = np.array(X), np.array(Y)
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    yhat = X @ coef
    r2 = 1 - np.sum((Y - yhat) ** 2) / np.sum((Y - Y.mean()) ** 2)
    fits["pooled"] = dict(exp_lam=float(coef[0]), exp_batch=float(coef[1]),
                          const=float(coef[2]), r2=float(r2))
    print("\n  POOLED FIT  excess_leak ~ C * lam^a * batch^b :")
    print(f"    a (lam exponent)   = {coef[0]:+.3f}   (predicted +0.5)")
    print(f"    b (batch exponent) = {coef[1]:+.3f}   (predicted -0.5)")
    print(f"    r2 = {r2:.3f}")

    out = dict(grid=grid, excess=excess, fits=fits,
               config=dict(steps=STEPS, measure_tail=MEASURE_TAIL,
                           n_int=N_INT, n_bnd_pool=N_BND_POOL, seeds=SEEDS,
                           lams=LAMS, batches=BATCHES),
               wall_s=round(time.time() - t0, 1))
    OUT_JSON.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT_JSON}  ({out['wall_s']}s total)")


if __name__ == "__main__":
    main()
