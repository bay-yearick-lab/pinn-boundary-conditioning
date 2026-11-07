# Reproduce the paper's results. PYTHON can point at any interpreter with the
# dependencies from pyproject.toml, e.g. `make verify PYTHON=.venv/bin/python`.
PYTHON ?= python
RUN := PYTHONPATH=src $(PYTHON)
# The dense problems are small; a few BLAS threads avoid oversubscription.
export OMP_NUM_THREADS ?= 4
export OPENBLAS_NUM_THREADS ?= 4

.PHONY: all verify verify-slow test headline experiments experiments-neural \
        experiments-gpu figures assets benchmarks clean

all: verify figures

# Fast core (under 6 min on a 4-thread CPU): exact linear laws, coupled-schedule
# and minibatch extensions, held-out and 3D a-priori diagnostics, then the
# headline numbers recomputed from data/.
verify:
	$(RUN) src/verify_statics.py
	$(RUN) src/verify_dynamics.py
	$(RUN) src/verify_ushape.py
	$(RUN) src/verify_exact_laws.py
	$(RUN) src/verify_extensions.py
	$(RUN) src/verify_coupled_path.py
	$(RUN) src/exp_diagnostic_holdout.py
	$(RUN) src/exp_diagnostic_3d.py --out build/diagnostic_3d.json
	$(PYTHON) scripts/headline.py

# Adds the nonlinear stationary-point check on a tanh network (needs PyTorch).
verify-slow: verify
	$(RUN) src/verify_nonlinear.py

test:
	$(PYTHON) -m pytest -q tests

headline:
	$(PYTHON) scripts/headline.py

# Fixed-feature figure data (CPU) and the 2D neural sweeps (CUDA when available).
# Sweeps append to data/*.jsonl and skip runs already recorded there.
experiments:
	$(RUN) src/exp_linear_frontier.py
	$(RUN) src/exp_ladder.py
	$(RUN) src/exp_precision_floor.py
	$(RUN) src/exp_pinn_sweep.py --seeds 5
	$(RUN) src/exp_pinn_sweep.py --seeds 5 --ledger
	$(RUN) src/exp_pinn_sweep.py --seeds 5 --pde helmholtz
	$(RUN) src/exp_pinn_sweep.py --seeds 5 --pde helmholtz --ledger
	$(RUN) src/exp_pinn_adaptive.py
	$(RUN) src/exp_budget_sweep.py
	$(RUN) src/exp_lbfgs_column.py
	$(RUN) src/exp_burgers.py
	$(RUN) src/exp_advection.py
	$(RUN) src/exp_heat_ic.py
	$(RUN) src/exp_operator.py

experiments-neural:
	$(RUN) src/exp_ntk_constants.py
	$(RUN) src/exp_ntk_drift.py
	$(RUN) src/exp_prospective_trained.py
	$(RUN) src/exp_minibatch_noise.py
	$(RUN) src/exp_schedule_ablation.py

# Large 3D sweep, 6x128 tanh network on a CUDA GPU (about 30 min).
experiments-gpu:
	$(RUN) src/exp_trained_3d.py

figures:
	$(PYTHON) scripts/make_figures.py

assets:
	$(PYTHON) tools/make_readme_assets.py

benchmarks:
	$(PYTHON) benchmarks/run_benchmarks.py


clean:
	rm -rf build .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
