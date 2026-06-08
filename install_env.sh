#!/usr/bin/env bash
# install_env.sh -- conditioned Audio DiT environment setup (IRCAM Linux)
# =====================================================================
# Assumes the venv is ALREADY created on top of tf2.18 and ACTIVE:
#
#   conda activate tf2.18
#   python -m venv --system-site-packages /data/anasynth_nonbp/baione/envs/genaudio_cond_env
#   source /data/anasynth_nonbp/baione/envs/genaudio_cond_env/bin/activate
#   bash install_env.sh
#
# What it does, and WHY (lessons learned setting this up on gaogu/guzheng):
#   1. pip install -r requirements_ircam.txt
#        Everything on top of the tf2.18 base. protobuf is NOT pinned in that
#        file: descript-audiotools declares protobuf<3.20, which pip cannot
#        reconcile with the protobuf==4.25.9 that CLAP/FAD need -> ResolutionImpossible.
#   2. pip install protobuf==4.25.9
#        Forced AFTER the requirements. A non-blocking warning
#        ("descript-audiotools requires protobuf<3.20 but you have 4.25.9") is
#        EXPECTED and ignored: the combination works in practice.
#   3. pip uninstall -y wandb
#        Not used (logging is TensorBoard). It also pulls an old protobuf and an
#        old wandb pin; removing it keeps the env clean.
#   4. import checks (must all print OK).
#
# torch is NOT installed here: it comes from the tf2.18 base (CUDA 12.x).
# =====================================================================

set -u  # error on unset vars; do NOT use -e (we tolerate the protobuf warning)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ="${HERE}/requirements_ircam.txt"

# --- guard: a venv must be active ---
if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "[install_env] ERROR: no active venv. Activate it first:"
    echo "  conda activate tf2.18"
    echo "  source /data/anasynth_nonbp/baione/envs/genaudio_cond_env/bin/activate"
    exit 1
fi
echo "[install_env] venv: ${VIRTUAL_ENV}"
echo "[install_env] python: $(which python)"

if [ ! -f "${REQ}" ]; then
    echo "[install_env] ERROR: ${REQ} not found."
    exit 1
fi

echo
echo "[install_env] 1/4  pip install -r requirements_ircam.txt"
pip install -r "${REQ}"

echo
echo "[install_env] 2/4  forcing protobuf==4.25.9 (ignore the descript-audiotools warning)"
pip install "protobuf==4.25.9"

echo
echo "[install_env] 3/4  removing wandb (not used; pulls old protobuf)"
pip uninstall -y wandb || true
# wandb removal can re-touch protobuf; make sure 4.25.9 stays in place
pip install "protobuf==4.25.9" >/dev/null 2>&1 || true

echo
echo "[install_env] 4/4  import checks"
python - <<'PYCHECK'
import sys, os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
ok = True
def check(label, fn):
    global ok
    try:
        fn(); print(f"  [OK]   {label}")
    except Exception as e:
        ok = False; print(f"  [FAIL] {label}: {type(e).__name__}: {e}")

import google.protobuf
print(f"  protobuf {google.protobuf.__version__} (expected 4.25.9)")
check("numpy + torch",        lambda: __import__("numpy") and __import__("torch"))
check("dac + encodec",        lambda: __import__("dac") and __import__("encodec"))
check("basic_pitch",          lambda: __import__("basic_pitch"))
check("beat_this",            lambda: __import__("beat_this"))
check("mir_eval",             lambda: __import__("mir_eval"))
check("laion_clap",           lambda: __import__("laion_clap"))
check("frechet_audio_distance", lambda: __import__("frechet_audio_distance"))
check("project modules", lambda: [__import__(m) for m in
       ("conditions","network_cond","audio_dataset_cond","metrics","condition_metrics")])
print()
print("[install_env] ALL OK" if ok else "[install_env] SOME CHECKS FAILED -- see above")
sys.exit(0 if ok else 1)
PYCHECK

echo
echo "[install_env] done."
