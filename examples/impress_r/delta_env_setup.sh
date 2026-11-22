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
#   export IMPRESS_USECASE=protein_binding   # or small_molecule_binding,
#                                            # or interface_targeting
#   bash delta_env_setup.sh
#
# IMPRESS_USECASE selects the default venv:
#   protein_binding      → $WORK_DIR/ve/impress   (IMPRESS's impress_A venv)
#   small_molecule_binding → $WORK_DIR/ve/small_mol
#   interface_targeting    → $WORK_DIR/ve/small_mol  (+ itgt from $ITGT_DIR)
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
#   ASTEER_DIR          asteer source tree   (default: $WORK_DIR/autoagent_steered_diffusion)
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
    interface_targeting)
        # Same pipeline and therefore the same venv as small_molecule_binding;
        # this use case adds the itgt surrogate on top rather than changing the
        # toolchain underneath.
        _DEFAULT_ENV="${WORK_DIR}/ve/small_mol"
        _IMPRESS_SETUP="IMPRESS/examples/small_molecule_binding/delta_env_setup.sh"
        _INSTALL_ITGT="yes"
        ;;
    *)
        echo "ERROR: Unknown IMPRESS_USECASE=${IMPRESS_USECASE}"
        echo "       Valid: protein_binding, small_molecule_binding, interface_targeting"
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

# ── 2a. asteer (the ASD seam) ─────────────────────────────────────────────────
# The ASD surrogate example (examples/impress/small_molecule_binding/) takes its
# corpus from asteer.labels rather than from DataManager's watermark: a sampler
# selects a non-contiguous subset and no watermark value describes one. asteer
# declares no dependencies, so this costs nothing to install and nothing to
# import. A WARNING rather than an error, because every other pipeline here runs
# without it.
echo ""
echo "── Step 2a: asteer (ASD contracts and label queue) ──"
ASTEER_DIR="${ASTEER_DIR:-${WORK_DIR}/autoagent_steered_diffusion}"
if [ -d "${ASTEER_DIR}" ]; then
    "${PIP}" install -q -e "${ASTEER_DIR}"
    echo "  asteer installed from ${ASTEER_DIR}"
else
    echo "  [WARN] ASTEER_DIR not found: ${ASTEER_DIR}"
    echo "         examples/impress/small_molecule_binding/ will not import, and"
    echo "         its tests skip rather than fail. Clone it there or export"
    echo "         ASTEER_DIR."
fi

# ── 2b. itgt (interface_targeting only) ───────────────────────────────────────
if [ "${_INSTALL_ITGT:-}" = "yes" ]; then
    echo ""
    echo "── Step 2b: itgt (interface-targeting models) ──"
    ITGT_DIR="${ITGT_DIR:-${WORK_DIR}/interface-targeting}"
    if [ -d "${ITGT_DIR}" ]; then
        # '[ml]' adds torch and scikit-learn. The default surrogate
        # (reward_ridge) needs only scikit-learn, and the two halves of the
        # extra fail independently, so install both and let the component
        # choose.
        "${PIP}" install -q -e "${ITGT_DIR}[ml]" \
            || "${PIP}" install -q -e "${ITGT_DIR}"
        echo "  itgt installed from ${ITGT_DIR}"
    else
        echo "  [WARN] ITGT_DIR not found: ${ITGT_DIR}"
        echo "         The campaign will run with a frozen diffusion target."
        echo "         Clone interface-targeting there or export ITGT_DIR."
    fi
fi

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
if [ -d "${ASTEER_DIR}" ]; then
    _check "asteer" "${PY}" -c "import asteer.labels; print('ok')"
fi
if [ "${_INSTALL_ITGT:-}" = "yes" ]; then
    _check "itgt" "${PY}" -c "import itgt; print('ok')"
fi

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
