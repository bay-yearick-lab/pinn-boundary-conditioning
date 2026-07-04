"""Error versus optimizer budget: does the fence's lead persist, grow, or close?

Three arms on the 2D Poisson MLP problem of exp_pinn_sweep.py, at Adam budgets
T in {5000, 80000} (the 20000 column already lives in pinn_sweep.jsonl):

  soft lam=100 : the best fixed weight found by the full sweep at T=20000
  hard         : transfinite-interpolation lifting (the fence)
  ledger lam=100 : the multiplier-corrected loss (the ledger)

Five seeds per arm per budget. Cosine LR schedule scales with T, as in the
main sweep, so each budget is a self-consistent training run rather than a
checkpoint of a longer one. Appends to data/budget_sweep.jsonl.
"""

import json
from pathlib import Path

import torch

from exp_pinn_sweep import run_one

OUT = Path(__file__).resolve().parent.parent / "data" / "budget_sweep.jsonl"

BUDGETS = (5000, 80000)
ARMS = (("soft", 1e2), ("hard", None), ("ledger", 1e2))
SEEDS = 5


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            r = json.loads(line)
            done.add((r["mode"], r["lam"], r["seed"], r["steps"]))
    with OUT.open("a") as fh:
        for steps in BUDGETS:
            for mode, lam in ARMS:
                for seed in range(SEEDS):
                    if (mode, lam, seed, steps) in done:
                        continue
                    rec = run_one(mode, lam, seed, steps, device, torch.float32)
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    print(f"T={steps} {mode:7s} lam={str(lam):6s} seed={seed} "
                          f"rel_l2={rec['rel_l2']:.3e} bnd={rec['bnd_rms']:.3e} "
                          f"({rec['wall_s']}s)")


if __name__ == "__main__":
    main()
