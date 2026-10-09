#!/bin/bash
# Wrapper around sbatch that routes logs to <use_case>/logs/ via --job-name.
# Usage: bash submit.sh [extra sbatch args]
#   export IMPRESS_USECASE=protein_binding   # or small_molecule_binding
#   export WORK_DIR=/work/nvme/bdyk/$USER
#   export SBATCH_ACCOUNT=<project>-delta-gpu
USECASE="${IMPRESS_USECASE:-protein_binding}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${SCRIPT_DIR}/${USECASE}/logs"
exec sbatch --job-name="${USECASE}" "$@" "${SCRIPT_DIR}/delta_gpu_run.sh"
