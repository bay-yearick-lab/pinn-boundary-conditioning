#!/usr/bin/env sh
# One-command verification, identical to `make verify`, for systems without make.
# Usage: sh scripts/verify.sh            (uses `python` on PATH)
#        PYTHON=.venv/bin/python sh scripts/verify.sh
set -e
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
for s in verify_statics verify_dynamics verify_ushape verify_exact_laws \
         verify_extensions verify_coupled_path exp_diagnostic_holdout; do
  echo "=== $s"
  "$PY" "src/$s.py"
done
echo "=== exp_diagnostic_3d"
"$PY" src/exp_diagnostic_3d.py --out build/diagnostic_3d.json
"$PY" scripts/headline.py
