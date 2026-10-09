#!/bin/bash
# =============================================================================
# IMPRESS-R ROME — unified environment setup
#
# Adds ROME-A to an existing IMPRESS venv created by the corresponding
# IMPRESS delta_env_setup.sh.  Run that script first; this one only
# installs ROME on top.
#
# Usage:
#   export WORK_DIR=/work/nvme/bdyk/$USER
#   export IMPRESS_USECASE=protein_binding   # or small_molecule_binding
#   bash delta_env_setup.sh
#
# IMPRESS_USECASE selects the default venv:
#   protein_binding      → $WORK_DIR/ve/impress   (IMPRESS's impress_A venv)
#   small_molecule_binding → $WORK_DIR/ve/small_mol
#
# Prerequisites:
#   - The matching IMPRESS delta_env_setup.sh already completed
#   - ROME source tree cloned to $WORK_DIR/ROME
#   - Internet access (login nodes have it; compute nodes do not)
#
# Optional overrides (CLI args):
#   --env-dir     DIR   venv location       (overrides use-case default)
#   --impress-dir DIR   IMPRESS source tree  (default: $WORK_DIR/IMPRESS)
#   --rome-dir    DIR   ROME source tree     (default: $WORK_DIR/ROME)
# =============================================================================
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail
fi

# ── Work directory ────────────────────────────────────────────────────────────
: "${WORK_DIR:?Set WORK_DIR before running, e.g.: export WORK_DIR=/work/nvme/bdyk/\$USER}"

# ── Use-case default venv ─────────────────────────────────────────────────────
IMPRESS_USECASE="${IMPRESS_USECASE:-protein_binding}"
case "${IMPRESS_USECASE}" in
    protein_binding)
        _DEFAULT_ENV="${WORK_DIR}/ve/impress"
        _IMPRESS_SETUP="IMPRESS/examples/protein_binding/delta_env_setup.sh"
        ;;
    small_molecule_binding)
        _DEFAULT_ENV="${WORK_DIR}/ve/small_mol"
        _IMPRESS_SETUP="IMPRESS/examples/small_molecule_binding/delta_env_setup.sh"
        ;;
    *)
        echo "ERROR: Unknown IMPRESS_USECASE=${IMPRESS_USECASE}"
        echo "       Valid: protein_binding, small_molecule_binding"
        exit 1
        ;;
esac

# ── Defaults / arg parsing ────────────────────────────────────────────────────
ENV_DIR="${ENV_DIR:-${_DEFAULT_ENV}}"
IMPRESS_DIR="${IMPRESS_DIR:-${WORK_DIR}/IMPRESS}"
ROME_DIR="${ROME_DIR:-${WORK_DIR}/ROME}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --env-dir)      ENV_DIR="$2";      shift 2 ;;
        --impress-dir)  IMPRESS_DIR="$2";  shift 2 ;;
        --rome-dir)     ROME_DIR="$2";     shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

PY="${ENV_DIR}/bin/python"
PIP="${ENV_DIR}/bin/pip"

echo "================================================================="
echo "  IMPRESS_USECASE    = ${IMPRESS_USECASE}"
echo "  WORK_DIR           = ${WORK_DIR}"
echo "  ENV_DIR            = ${ENV_DIR}"
echo "  IMPRESS_DIR        = ${IMPRESS_DIR}"
echo "  ROME_DIR           = ${ROME_DIR}"
echo "================================================================="

# ── 1. Verify base venv exists ────────────────────────────────────────────────
echo ""
echo "── Step 1: Checking base venv ──"
if [ ! -x "${PY}" ]; then
    echo "ERROR: venv not found at ${ENV_DIR}"
    echo "       Run \${WORK_DIR}/${_IMPRESS_SETUP} first."
    exit 1
fi
echo "  Base venv OK: $("${PY}" --version)"

# ── 2. ROME ───────────────────────────────────────────────────────────────────
echo ""
echo "── Step 2: ROME (editable) ──"
if [ ! -d "${ROME_DIR}" ]; then
    echo "ERROR: ROME_DIR not found: ${ROME_DIR}"
    echo "       Clone ROME and re-run, or pass --rome-dir /path/to/ROME"
    exit 1
fi
"${PIP}" install -q -e "${ROME_DIR}"
echo "  ROME installed from ${ROME_DIR}"

# ── 3. Verify ─────────────────────────────────────────────────────────────────
echo ""
echo "── Step 3: Verifying ──"
_check() {
    local label="$1"; shift
    if out=$("$@" 2>&1); then
        echo "  [OK] ${label}: ${out}"
    else
        echo "  [WARN] ${label} failed:"
        echo "    ${out}" | head -3
    fi
}
_check "impress" "${PY}" -c "import impress; print('ok')"
_check "rome"    "${PY}" -c "import rome; print('ok')"

echo ""
echo "================================================================="
echo "ROME addon complete."
echo ""
echo "Submit the pipeline:"
echo "  export SBATCH_ACCOUNT=<your-project>-delta-gpu"
echo "  export WORK_DIR=${WORK_DIR}"
echo "  export IMPRESS_USECASE=${IMPRESS_USECASE}"
echo "  cd ${ROME_DIR}/examples/impress_r"
echo "  mkdir -p logs && sbatch delta_gpu_run.sh"
echo ""
echo "Smoke test (no GPU needed for training):"
echo "  ROME_TRAINER=dummy sbatch delta_gpu_run.sh"
echo "================================================================="
