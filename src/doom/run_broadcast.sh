#!/usr/bin/env bash
# Launch the existing, reviewed model without BLAS thread oversubscription.
# This does not restart or connect to an already-running broadcaster.
set -euo pipefail
DOOMFLY_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DOOMFLY_PROJECT_ROOT"

# The retinal transform contains tiny three-channel matrix/vector products.
# OpenBLAS worker spin-waiting competes with the single-threaded neural kernel.
# Set this before Python imports NumPy; no equations, graph or timestep change.
export OPENBLAS_NUM_THREADS=1
exec "$DOOMFLY_PROJECT_ROOT/.venv-neural/bin/python" -m doom.server "$@"
