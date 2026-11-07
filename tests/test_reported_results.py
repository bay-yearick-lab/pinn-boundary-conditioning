"""The headline numbers in README.md, recomputed from the shipped run records."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"


def medians(name, key="rel_l2", **filt):
    groups = defaultdict(list)
    for line in (DATA / name).read_text().splitlines():
        r = json.loads(line)
        if all(r.get(k) == v for k, v in filt.items()):
            groups[(r["mode"], r.get("lam"))].append(r[key])
    return {k: float(np.median(v)) for k, v in groups.items()}


def test_poisson_pinn_valley_and_hard_twin():
    m = medians("pinn_sweep.jsonl", steps=20000, dtype="float32")
    assert round(m[("soft", 0.01)], 2) == 0.37
    best = min(v for (mode, _), v in m.items() if mode == "soft")
    assert f"{best:.1e}" == "1.2e-03"
    assert f"{m[('hard', None)]:.1e}" == "1.3e-04"
    assert 8.5 < best / m[("hard", None)] < 9.5


def test_physics_informed_deeponet():
    m = medians("operator_sweep.jsonl", key="rel_l2_med", steps=15000)
    best = min(v for (mode, _), v in m.items() if mode == "soft")
    assert f"{best:.1e}" == "7.1e-03"
    assert f"{m[('hard', None)]:.1e}" == "7.8e-04"


def test_burgers():
    m = medians("burgers_sweep.jsonl", steps=20000)
    best = min(v for (mode, _), v in m.items() if mode == "soft")
    assert f"{best:.1e}" == "5.6e-04"
    assert f"{m[('hard', None)]:.1e}" == "3.0e-05"


def test_3d_product_limit_prediction():
    d = json.loads((DATA / "diagnostic_3d.json").read_text())
    s = {r["testbed"]: r for r in d["summary"]}
    assert f"{100 * s['3D Poisson']['product_relerr']['median']:.2f}" == "0.36"
    assert f"{100 * s['3D stiff anisotropic']['product_relerr']['median']:.2f}" == "0.22"
    assert all(r["leak_brackets_all_hold"] for r in d["summary"])


def test_gpu_trained_3d():
    d = json.loads((DATA / "trained_3d.json").read_text())
    p = d["P1_poisson"]["confirmation"]
    assert f"{p['best_soft_median_vs_fd']:.1e}" == "6.2e-02"
    assert f"{p['hard_median_vs_fd']:.1e}" == "5.5e-03"
    assert p["predicted_valley_in_observed_region"]
    s = d["P2_stiff"]["confirmation"]
    assert f"{s['hard_median_vs_fd']:.1e}" == "4.7e-02"
