# Data records

Run records written by the scripts in `src/`. Every file is small (largest
0.55 MB) and is read by `scripts/make_figures.py`, `scripts/headline.py`,
`tools/make_readme_assets.py` and `tests/test_reported_results.py`.

| File | Written by | Contents |
| --- | --- | --- |
| `linear_frontier.json` | `exp_linear_frontier.py` | leak, conditioning valley, budget frontier (fixed features) |
| `ladder.json` | `exp_ladder.py` | penalty vs augmented Lagrangian vs consistent penalty vs lifting |
| `precision_floor.json` | `exp_precision_floor.py` | float32 / float64 floor of exact penalized solves |
| `pinn_sweep.jsonl` | `exp_pinn_sweep.py` | 2D Poisson tanh PINN: weight sweep, hard twin, multiplier arm, float64 control |
| `pinn_sweep_helmholtz.jsonl` | `exp_pinn_sweep.py --pde helmholtz` | indefinite Helmholtz, k^2 = 40 |
| `pinn_fp64_control.jsonl` | `exp_pinn_sweep.py` | float64 control trajectories |
| `pinn_adaptive.jsonl` | `exp_pinn_adaptive.py` | gradient-statistic and kernel-trace weighting rules |
| `budget_sweep.jsonl` | `exp_budget_sweep.py` | error vs optimizer budget (5k, 80k steps) |
| `lbfgs_column.jsonl` | `exp_lbfgs_column.py` | Adam then L-BFGS, float32 |
| `burgers_sweep.jsonl` | `exp_burgers.py` | forced viscous Burgers |
| `advection_sweep.jsonl` | `exp_advection.py` | advection-dominated boundary layer |
| `heat_ic_sweep.jsonl` | `exp_heat_ic.py` | soft vs exact initial condition |
| `operator_sweep.jsonl` | `exp_operator.py` | physics-informed DeepONet, 10 seeds |
| `ntk_constants.json`, `ntk_drift.json` | `exp_ntk_constants.py`, `exp_ntk_drift.py` | tangent-kernel constants of the trained network |
| `prospective_trained.json` | `exp_prospective_trained.py` | constants committed before the confirming sweep |
| `minibatch_noise.json(l)` | `exp_minibatch_noise.py` | stochastic leak floor vs weight and batch size |
| `schedule_ablation.json(l)` | `exp_schedule_ablation.py` | weight schedules with a shared endpoint |
| `diagnostic_3d.json` | `exp_diagnostic_3d.py` | 3D a-priori constants vs direct-solve sweep, 10 seeds |
| `trained_3d.json` | `exp_trained_3d.py` | GPU-trained 6x128 3D PINN sweep, hard twin, FD reference |

Every `.jsonl` file appends one JSON record per completed run. The unique key
of a record is the full tuple `(mode, lam, seed, steps, dtype)` where present;
some files intentionally contain rows that share `(mode, lam, seed)`:

- `pinn_sweep.jsonl`: the float64 control reruns three soft configurations at
  `seed=0` with `dtype="float64"` alongside the float32 sweep.
- `budget_sweep.jsonl`: each arm appears at `steps=5000` and `steps=80000`;
  the `steps=20000` column lives in `pinn_sweep.jsonl`.

Filter on the full key, never on `(mode, lam, seed)` alone.
