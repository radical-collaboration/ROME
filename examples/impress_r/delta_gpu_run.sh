#!/bin/bash
#
# IMPRESS-R ROME — unified SLURM batch script (Delta HPC / GPU)
#
# Use submit.sh to launch (it sets --job-name so logs go to <case>/logs/):
#   cd $WORK_DIR/ROME/examples/impress_r
#   export IMPRESS_USECASE=protein_binding   # or small_molecule_binding
#   bash submit.sh
#
# Required:
#   export SBATCH_ACCOUNT=<project>-delta-gpu
#   export WORK_DIR=/work/nvme/bdyk/$USER
#
# ── Shared ROME env vars ──────────────────────────────────────────────────────
#   ROME_DIR         — ROME checkout          (default: $WORK_DIR/ROME)
#   ROME_TRAINER     — mpnn                   (default: mpnn)
#   ROME_MIN_SAMPLES — corpus before 1st round (default: use-case specific)
#   ROME_FALLBACK    — DDict result grace s    (default: 120)
#   ROME_REWARD_FN   — module:fn override      (default: use-case specific)
#
# ── Use-case: protein_binding ─────────────────────────────────────────────────
#   MPNN_PATH        — ProteinMPNN checkout   (default: $WORK_DIR/ProteinMPNN)
#   BOLTZ_VENV       — Boltz venv             (default: $WORK_DIR/ve/boltz)
#   BOLTZ_CACHE_DIR  — Boltz weight cache     (default: ~/boltz)
#   IMPRESS_SCRIPTS_DIR — IMPRESS pb examples dir
#   IMPRESS_BASE_DIR    — parent of prod_in/
#   IMPRESS_OUTPUT_DIR  — campaign output root
#   IMPRESS_N_PIPELINES — top-level pipelines  (default: 1)
#
# ── Use-case: small_molecule_binding ─────────────────────────────────────────
#   MPNN_DIR         — LigandMPNN checkout    (default: $WORK_DIR/LigandMPNN)
#   BOLTZ_CACHE      — Boltz weight cache     (default: $WORK_DIR/.cache/boltz)
#   FOUNDRY_SIF_PATH — RFD3 singularity sandbox (or set FOUNDRY_TAR)
#   IMPRESS_DIR      — IMPRESS checkout       (default: $WORK_DIR/IMPRESS)
#   IMPRESS_OUTPUT_DIR  — campaign output root
#   IMPRESS_N_PIPELINES — top-level pipelines  (default: 2)
#
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --mem=220G
#SBATCH --time=06:00:00
#SBATCH --job-name=impress_rome
#SBATCH --mail-user=mgoliyad@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=%x/logs/impress_%j.out
#SBATCH --error=%x/logs/impress_%j.err

set -e

# ── Use-case selection ────────────────────────────────────────────────────────
IMPRESS_USECASE="${IMPRESS_USECASE:-protein_binding}"
case "${IMPRESS_USECASE}" in
    protein_binding|small_molecule_binding) ;;
    *)
        echo "ERROR: Unknown IMPRESS_USECASE=${IMPRESS_USECASE}"
        echo "       Valid: protein_binding, small_molecule_binding"
        exit 1
        ;;
esac

# ── Sanity checks ─────────────────────────────────────────────────────────────
if [ -z "${SBATCH_ACCOUNT:-}${SLURM_JOB_ACCOUNT:-}" ]; then
    echo "WARNING: SBATCH_ACCOUNT is not set — job may be charged to default account."
fi
echo "Account:      ${SLURM_JOB_ACCOUNT:-unknown}"
echo "Use-case:     ${IMPRESS_USECASE}"

if [ -z "${WORK_DIR:-}" ]; then
    echo "ERROR: WORK_DIR is not set."
    echo "       export WORK_DIR=/work/nvme/bdyk/\$USER && bash submit.sh"
    exit 1
fi

# ── System library paths (Delta-specific, required by Dragon) ─────────────────
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/cuda/12.8
export MPI_LIB=/opt/cray/pe/mpich/8.1.32/ofi/gnu/11.2/lib-abi-mpich
export FAB_LIB=/opt/cray/libfabric/1.22.0/lib64
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${MPI_LIB}:${FAB_LIB}:${LD_LIBRARY_PATH:-}

# ── Shared ROME settings ──────────────────────────────────────────────────────
export ROME_DIR="${ROME_DIR:-${WORK_DIR}/ROME}"
export ROME_TRAINER="${ROME_TRAINER:-mpnn}"
# 120 s gives training rounds time to finish before Dragon's DDict result
# delivery future blocks (dragonhpc 0.14.1 DDict race).
export ROME_FALLBACK="${ROME_FALLBACK:-120}"

export IMPRESS_BACKEND="${IMPRESS_BACKEND:-dragon}"
export IMPRESS_TEST_MODE="${IMPRESS_TEST_MODE:-0}"

# ── Use-case specific setup ───────────────────────────────────────────────────
if [ "${IMPRESS_USECASE}" = "protein_binding" ]; then

    IMPRESS_VENV="${IMPRESS_VENV:-${WORK_DIR}/ve/impress}"
    export MPNN_PATH="${MPNN_PATH:-${WORK_DIR}/ProteinMPNN}"
    export BOLTZ_VENV="${BOLTZ_VENV:-${WORK_DIR}/ve/boltz}"
    export BOLTZ_CACHE_DIR="${BOLTZ_CACHE_DIR:-${HOME}/boltz}"
    mkdir -p "${BOLTZ_CACHE_DIR}"
    export IMPRESS_SCRIPTS_DIR="${IMPRESS_SCRIPTS_DIR:-${WORK_DIR}/IMPRESS/examples/protein_binding}"
    export IMPRESS_BASE_DIR="${IMPRESS_BASE_DIR:-${WORK_DIR}/IMPRESS_inputs}"
    export IMPRESS_OUTPUT_DIR="${IMPRESS_OUTPUT_DIR:-${ROME_DIR}/examples/impress_r/protein_binding/campaign_outputs}"
    export IMPRESS_N_PIPELINES="${IMPRESS_N_PIPELINES:-1}"
    export ROME_MIN_SAMPLES="${ROME_MIN_SAMPLES:-4}"
    ROME_SCRIPT="run_protein_binding_rome.py"

    if [ ! -d "${MPNN_PATH}" ]; then
        echo "ERROR: MPNN_PATH does not exist: ${MPNN_PATH}"
        echo "       Clone dauparas/ProteinMPNN there or set MPNN_PATH."
        exit 1
    fi

    echo "MPNN_PATH:           ${MPNN_PATH}"
    echo "BOLTZ_VENV:          ${BOLTZ_VENV}"
    echo "IMPRESS_SCRIPTS_DIR: ${IMPRESS_SCRIPTS_DIR}"
    echo "IMPRESS_BASE_DIR:    ${IMPRESS_BASE_DIR}"

elif [ "${IMPRESS_USECASE}" = "small_molecule_binding" ]; then

    IMPRESS_VENV="${IMPRESS_VENV:-${WORK_DIR}/ve/small_mol}"
    export MPNN_DIR="${MPNN_DIR:-${WORK_DIR}/LigandMPNN}"
    export BOLTZ_VENV="${BOLTZ_VENV:-${WORK_DIR}/ve/boltz}"
    export BOLTZ_CACHE="${BOLTZ_CACHE:-${WORK_DIR}/.cache/boltz}"
    mkdir -p "${BOLTZ_CACHE}"
    export IMPRESS_DIR="${IMPRESS_DIR:-${WORK_DIR}/IMPRESS}"
    export IMPRESS_OUTPUT_DIR="${IMPRESS_OUTPUT_DIR:-${ROME_DIR}/examples/impress_r/small_molecule_binding/campaign_outputs}"
    export IMPRESS_N_PIPELINES="${IMPRESS_N_PIPELINES:-2}"
    export ROME_MIN_SAMPLES="${ROME_MIN_SAMPLES:-8}"
    ROME_SCRIPT="run_smb_rome.py"

    if [ ! -d "${MPNN_DIR}" ]; then
        echo "ERROR: MPNN_DIR does not exist: ${MPNN_DIR}"
        echo "       Clone dauparas/LigandMPNN there or set MPNN_DIR."
        exit 1
    fi

    # Foundry sandbox (RFD3): extract to /tmp, clean up on exit.
    if [ -z "${FOUNDRY_SIF_PATH:-}" ] && [ -f "${WORK_DIR}/foundry.sif" ]; then
        export FOUNDRY_SIF_PATH="${WORK_DIR}/foundry.sif"
    fi
    if [ -z "${FOUNDRY_SIF_PATH:-}" ]; then
        FOUNDRY_TAR="${FOUNDRY_TAR:-${WORK_DIR}/foundry_sandbox.tar.gz}"
        if [ ! -f "${FOUNDRY_TAR}" ]; then
            echo "ERROR: foundry sandbox tarball not found: ${FOUNDRY_TAR}"
            echo "       Build it first: sbatch \${IMPRESS_DIR}/examples/small_molecule_binding/pull_foundry.sh"
            exit 1
        fi
        _FOUNDRY_TMP="/tmp/foundry_${SLURM_JOB_ID:-$$}"
        echo "Extracting foundry sandbox to ${_FOUNDRY_TMP} ..."
        mkdir -p "${_FOUNDRY_TMP}"
        tar -xzf "${FOUNDRY_TAR}" -C "${_FOUNDRY_TMP}" --strip-components=1
        export FOUNDRY_SIF_PATH="${_FOUNDRY_TMP}"
        trap "echo 'Removing ${_FOUNDRY_TMP}'; rm -rf '${_FOUNDRY_TMP}'" EXIT
    fi

    echo "MPNN_DIR:            ${MPNN_DIR}"
    echo "BOLTZ_VENV:          ${BOLTZ_VENV}"
    echo "BOLTZ_CACHE:         ${BOLTZ_CACHE}"
    echo "FOUNDRY_SIF_PATH:    ${FOUNDRY_SIF_PATH}"
    echo "IMPRESS_DIR:         ${IMPRESS_DIR}"

fi

# ── Activate venv (after use-case sets IMPRESS_VENV) ─────────────────────────
unset SLURM_EXPORT_ENV
source "${IMPRESS_VENV}/bin/activate"
dragon-config add --ofi-runtime-lib="${FAB_LIB}"

# ── Common env summary ────────────────────────────────────────────────────────
echo "WORK_DIR:            ${WORK_DIR}"
echo "ROME_DIR:            ${ROME_DIR}"
echo "IMPRESS_OUTPUT_DIR:  ${IMPRESS_OUTPUT_DIR}"
echo "ROME_TRAINER:        ${ROME_TRAINER}"
echo "ROME_MIN_SAMPLES:    ${ROME_MIN_SAMPLES}"
echo "IMPRESS_BACKEND:     ${IMPRESS_BACKEND}"
echo "TEST_MODE:           ${IMPRESS_TEST_MODE}"

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR="${ROME_DIR}/examples/impress_r/${IMPRESS_USECASE}"
cd "${WORKDIR}"
mkdir -p "${IMPRESS_OUTPUT_DIR}"

# ── Run ───────────────────────────────────────────────────────────────────────
if [ "${SLURM_NNODES:-1}" -gt 1 ]; then
    DRAGON_MODE="-m"
else
    DRAGON_MODE="-s"
fi

rm -f ddict_orc*

echo "Running: dragon ${DRAGON_MODE} ${ROME_SCRIPT}  (nodes=${SLURM_NNODES:-1})"
dragon ${DRAGON_MODE} "${WORKDIR}/${ROME_SCRIPT}"
_rc=$?
dragon-cleanup-deprecated || true
rm -f ddict_orc*

echo "=== IMPRESS-R ROME [${IMPRESS_USECASE}] done (rc=${_rc}): $(date) ==="
exit ${_rc}
