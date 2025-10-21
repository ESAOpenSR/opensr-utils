#!/usr/bin/env bash
set -euo pipefail

# ================== CONFIG ==================
REPO_ROOT="/data1/simon/GitHub/opensr-utils"   # repo root (for PYTHONPATH if not installed)
DIR="/data2/simon/1msqkm/deep/tiles/"            # folder with inputs
EXT="zip"                                      # file extension to process

PYTHON="python"                                # interpreter
DEVICE="cuda"                                  # "cuda" or "cpu"
GPUS="0,1"                                 # comma list; ignored on CPU

WINDOW_H=128
WINDOW_W=128
FACTOR=4
OVERLAP=12
ELIM_BORDER=0
SAVE_PREVIEW=1      # 1=yes, 0=no
DEBUG=0             # 1=yes, 0=no
# ============================================

# Add repo to PYTHONPATH unless package is installed
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Make a real, re-executable Python runner for DDP children
TMPPY="${TMPDIR:-/tmp}/opensr_one.py"
cat > "${TMPPY}" <<'PY'
import os, sys, traceback
from pathlib import Path

def main():
    root = os.environ["SR_ROOT"]
    device = os.environ.get("SR_DEVICE","cuda")
    gpu_ids = [int(x) for x in os.environ.get("SR_GPUS","").split(",") if x.strip()]
    ws = (int(os.environ.get("SR_WS_H","128")), int(os.environ.get("SR_WS_W","128")))
    factor = int(os.environ.get("SR_FACTOR","4"))
    overlap = int(os.environ.get("SR_OVERLAP","8"))
    elim = int(os.environ.get("SR_ELIM_BORDER","0"))
    save_preview = os.environ.get("SR_SAVE_PREVIEW","0") == "1"
    debug = os.environ.get("SR_DEBUG","0") == "1"

    # Load lazily inside subprocess to keep runs isolated
    from opensr_utils.model_utils.get_models import get_ldsrs2
    from opensr_utils.pipeline import large_file_processing

    model = get_ldsrs2(device=device)
    gpus = gpu_ids if (device == "cuda" and gpu_ids) else None

    _ = large_file_processing(
        root=str(Path(root).resolve()),
        model=model,
        window_size=ws,
        factor=factor,
        overlap=overlap,
        eliminate_border_px=elim,
        device=device,
        gpus=gpus,
        save_preview=save_preview,
        debug=debug,
    )

if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
PY

chmod +x "${TMPPY}"

# Find files
shopt -s nullglob
mapfile -t files < <(printf '%s\n' "${DIR}"/*."${EXT}")
count=${#files[@]}
if (( count == 0 )); then
  echo "No *.${EXT} files found in ${DIR}"
  exit 0
fi

echo "Found ${count} *.${EXT} files in ${DIR}"

ok=0; fail=0; idx=0
for f in "${files[@]}"; do
  idx=$((idx+1))
  echo
  echo "[$idx/${count}] ▶️  Processing: $f"
  start_ts=$(date '+%Y-%m-%d %H:%M:%S')

  # Fresh DDP comms each run
  export MASTER_ADDR="127.0.0.1"
  export MASTER_PORT="$(shuf -i 29501-59999 -n 1)"

  # Env for the Python runner
  export SR_ROOT="$f"
  export SR_DEVICE="$DEVICE"
  export SR_GPUS="$GPUS"
  export SR_WS_H="$WINDOW_H"
  export SR_WS_W="$WINDOW_W"
  export SR_FACTOR="$FACTOR"
  export SR_OVERLAP="$OVERLAP"
  export SR_ELIM_BORDER="$ELIM_BORDER"
  export SR_SAVE_PREVIEW="$SAVE_PREVIEW"
  export SR_DEBUG="$DEBUG"

  # Run one file (sequential—wait for exit)
  if "${PYTHON}" "${TMPPY}"; then
    echo "   ✅ Done: $f  (started ${start_ts}, finished $(date '+%Y-%m-%d %H:%M:%S'))"
    ok=$((ok+1))
  else
    status=$?
    echo "   ❌ Failed: $f  (exit ${status})"
    # Exit code 137 commonly = OOM kill by OS. If you see it, try fewer GPUs or smaller batch/workers.
    fail=$((fail+1))
  fi
done

echo
echo "📊 Summary → OK: ${ok}  FAIL: ${fail}  TOTAL: ${count}"