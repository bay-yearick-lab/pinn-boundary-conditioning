"""Two published navigators on the MLP problem, implemented faithfully.

  anneal : the published gradient-statistic (learning-rate-annealing) rule: lam_hat = max|grad L_r| / mean|grad L_b| (unweighted
           denominator), EMA lam <- (1-alpha) lam + alpha lam_hat, alpha = 0.9,
           rebalanced every 100 steps.
  ntk    : the published tangent-kernel trace-balancing rule:
           lam = tr(K_rr)/tr(K_bb), traces estimated on subsampled points via
           per-sample parameter gradients, rebalanced every 500 steps.

Same problem, architecture, budget, and seeds as exp_pinn_sweep.py.
Appends to data/pinn_adaptive.jsonl.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

from exp_pinn_sweep import MLP, evaluate, f_rhs_t, laplacian, u_exact_t

OUT = Path(__file__).resolve().parent.parent / "data" / "pinn_adaptive.jsonl"


def trace_of_kernel(model, pts, fn):
    """tr(K) = sum_i ||grad_theta fn_i||^2 over the given points."""
    total = 0.0
    params = [p for p in model.parameters() if p.requires_grad]
    for i in range(pts.shape[0]):
        val = fn(pts[i:i + 1])
        grads = torch.autograd.grad(val, params, retain_graph=False,
                                    allow_unused=True)
        total += sum(float((g ** 2).sum()) for g in grads if g is not None)
    return total


def run_one(rule, seed, steps, device, dtype=torch.float32,
            n_int=2048, n_bnd=256, lr=1e-3, record_every=500):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP().to(device=device, dtype=dtype)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 1e-2)
    gen = torch.Generator(device="cpu").manual_seed(seed + 1)
    xy_int = torch.rand((n_int, 2), generator=gen).to(device=device, dtype=dtype)
    tb = torch.rand((n_bnd,), generator=gen).to(device=device, dtype=dtype)
    edge = (torch.arange(n_bnd) % 4).to(device)
    bx = torch.where(edge == 0, tb, torch.where(edge == 1, tb,
                     torch.where(edge == 2, torch.zeros_like(tb), torch.ones_like(tb))))
    by = torch.where(edge == 0, torch.zeros_like(tb), torch.where(edge == 1,
                     torch.ones_like(tb), torch.where(edge == 2, tb, tb)))
    xy_bnd = torch.stack([bx, by], dim=1)
    g_bnd = u_exact_t(xy_bnd[:, 0], xy_bnd[:, 1])
    f_int = f_rhs_t(xy_int[:, 0], xy_int[:, 1])

    lam_t = 1.0
    traj = []
    t0 = time.time()
    for step in range(steps + 1):
        opt.zero_grad(set_to_none=True)
        u_i, lap = laplacian(model, xy_int.clone(), hard=False)
        res = lap + f_int
        loss_r = torch.mean(res ** 2)
        u_b = model(xy_bnd).squeeze(-1)
        loss_b = torch.mean((u_b - g_bnd) ** 2)

        if rule == "anneal_w" and step % 100 == 0 and step > 0:
            # weighted-denominator convention: divide by the gradient of the
            # WEIGHTED boundary term; self-stabilizing fixed point
            gr = torch.autograd.grad(loss_r, list(model.parameters()),
                                     retain_graph=True, allow_unused=True)
            gb = torch.autograd.grad(loss_b, list(model.parameters()),
                                     retain_graph=True, allow_unused=True)
            mg = max(g.abs().max().item() for g in gr if g is not None)
            mb = np.mean([g.abs().mean().item() for g in gb if g is not None])
            if mb > 0 and lam_t > 0:
                lam_hat = mg / (lam_t * mb)     # fixed point: lam* = sqrt(mg/mb)
                lam_t = 0.1 * lam_t + 0.9 * lam_hat
        if rule == "anneal" and step % 100 == 0 and step > 0:
            gr = torch.autograd.grad(loss_r, list(model.parameters()),
                                     retain_graph=True, allow_unused=True)
            gb = torch.autograd.grad(loss_b, list(model.parameters()),
                                     retain_graph=True, allow_unused=True)
            mg = max(g.abs().max().item() for g in gr if g is not None)
            mb = np.mean([g.abs().mean().item() for g in gb if g is not None])
            if mb > 0:
                lam_hat = mg / mb
                lam_t = 0.1 * lam_t + 0.9 * lam_hat        # alpha = 0.9 (paper)
        elif rule == "ntk" and step % 500 == 0:
            k = 64
            sub_i = xy_int[torch.randperm(n_int)[:k]].clone()
            sub_b = xy_bnd[torch.randperm(n_bnd)[:k]].clone()

            def res_fn(p):
                _, lp = laplacian(model, p, hard=False)
                return (lp + f_rhs_t(p[:, 0], p[:, 1])).squeeze()

            def bnd_fn(p):
                return (model(p).squeeze(-1) - u_exact_t(p[:, 0], p[:, 1])).squeeze()

            tr_r = trace_of_kernel(model, sub_i, res_fn) * (n_int / k)
            tr_b = trace_of_kernel(model, sub_b, bnd_fn) * (n_bnd / k)
            if tr_b > 0:
                lam_t = tr_r / tr_b
            # recreate the graph consumed by the trace probes
            opt.zero_grad(set_to_none=True)
            u_i, lap = laplacian(model, xy_int.clone(), hard=False)
            res = lap + f_int
            loss_r = torch.mean(res ** 2)
            u_b = model(xy_bnd).squeeze(-1)
            loss_b = torch.mean((u_b - g_bnd) ** 2)

        loss = loss_r + lam_t * loss_b
        loss.backward()
        opt.step()
        sched.step()
        if step % record_every == 0:
            rel_l2, bnd_rms = evaluate(model, device, dtype)
            traj.append(dict(step=step, rel_l2=rel_l2, bnd_rms=bnd_rms,
                             lam=lam_t))
    rel_l2, bnd_rms = evaluate(model, device, dtype)
    return dict(mode=rule, seed=seed, steps=steps, dtype="float32",
                rel_l2=rel_l2, bnd_rms=bnd_rms, lam_final=lam_t,
                wall_s=round(time.time() - t0, 1), traj=traj)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    steps, seeds = 20000, 5
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            r = json.loads(line)
            done.add((r["mode"], r["seed"]))
    with OUT.open("a") as fh:
        for rule in ("anneal", "ntk", "anneal_w"):
            for seed in range(seeds):
                if (rule, seed) in done:
                    continue
                rec = run_one(rule, seed, steps, device)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print(f"{rule:7s} seed={seed} rel_l2={rec['rel_l2']:.3e} "
                      f"bnd={rec['bnd_rms']:.3e} lam_final={rec['lam_final']:.3g} "
                      f"({rec['wall_s']}s)")


if __name__ == "__main__":
    main()
