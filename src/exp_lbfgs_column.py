"""The quasi-Newton column: Adam then full-batch L-BFGS on the Poisson sweep.

The momentum-family theorem exempts quasi-Newton methods from the momentum ceiling; the
arithmetic floor kappa(lambda) eps_mach of the paper predicts that in float32 their gains
are capped by the kappa(lambda) eps_mach branch, i.e. the boundary error
should stop improving (and eventually worsen) as the weight grows, unlike
the flat Adam budget floor. This script measures that column.

Protocol: identical problem/architecture to exp_pinn_sweep (Poisson), Adam
10k steps then L-BFGS (full batch, strong Wolfe) 60 outer rounds x 20
iterations in float32, two seeds, lam in {1e0..1e5}.

Writes data/lbfgs_column.jsonl.
"""

import json
import time
from pathlib import Path

import torch

from exp_pinn_sweep import MLP, evaluate, f_rhs_t, laplacian, u_exact_t

OUT = Path(__file__).resolve().parent.parent / "data" / "lbfgs_column.jsonl"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def run_one(lam, seed, adam_steps=10000, lbfgs_rounds=60, lbfgs_iter=20):
    torch.manual_seed(seed)
    model = MLP().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(seed + 1)
    xy_int = torch.rand((2048, 2), generator=gen).to(DEV)
    tb = torch.rand((256,), generator=gen).to(DEV)
    edge = (torch.arange(256) % 4).to(DEV)
    zx, ox = torch.zeros_like(tb), torch.ones_like(tb)
    bx = torch.where(edge == 0, tb, torch.where(edge == 1, tb,
                     torch.where(edge == 2, zx, ox)))
    by = torch.where(edge == 0, zx, torch.where(edge == 1, ox,
                     torch.where(edge == 2, tb, tb)))
    xy_bnd = torch.stack([bx, by], dim=1)
    g_bnd = u_exact_t(xy_bnd[:, 0], xy_bnd[:, 1])
    f_int = f_rhs_t(xy_int[:, 0], xy_int[:, 1])

    def loss_fn():
        u_i, lap = laplacian(model, xy_int.clone(), hard=False)
        res = lap + f_int
        u_b = model(xy_bnd).squeeze(-1)
        return torch.mean(res ** 2) + lam * torch.mean((u_b - g_bnd) ** 2)

    t0 = time.time()
    for _ in range(adam_steps):
        opt.zero_grad(set_to_none=True)
        loss_fn().backward()
        opt.step()
    for _ in range(lbfgs_rounds):
        opt2 = torch.optim.LBFGS(model.parameters(), max_iter=lbfgs_iter,
                                 history_size=40, line_search_fn="strong_wolfe",
                                 tolerance_grad=0, tolerance_change=0)

        def closure():
            opt2.zero_grad(set_to_none=True)
            L = loss_fn()
            L.backward()
            return L
        opt2.step(closure)
    rel_l2, bnd_rms = evaluate(model, DEV, torch.float32)
    return dict(mode="lbfgs", lam=lam, seed=seed, rel_l2=rel_l2,
                bnd_rms=bnd_rms, wall_s=round(time.time() - t0, 1))


def main():
    OUT.parent.mkdir(exist_ok=True)
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            r = json.loads(line)
            done.add((r["lam"], r["seed"]))
    with OUT.open("a") as fh:
        for seed in (0, 1):
            for lam in (1e0, 1e1, 1e2, 1e3, 1e4, 1e5):
                if (lam, seed) in done:
                    continue
                rec = run_one(lam, seed)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print(f"lbfgs lam={lam:8g} seed={seed} rel={rec['rel_l2']:.3e} "
                      f"bnd={rec['bnd_rms']:.3e} ({rec['wall_s']}s)")


if __name__ == "__main__":
    main()
