#!/usr/bin/env bash
# Sequentially run SR on every file in a directory using a fresh Python process each time.

# ---------- CONFIG (edit these) ----------
DIR="/data2/simon/mosaic/inference"   # folder with inputs
EXT="zip"                              # file extension to process
PYTHON="python"                        # interpreter (or full path)
DEVICE="cuda"                          # "cuda" or "cpu"
GPUS="0,1,2,3"                         # comma list; ignored on CPU

WINDOW_H=128
WINDOW_W=128
FACTOR=4
OVERLAP=12
ELIM_BORDER=0
SAVE_PREVIEW=1      # 1=yes, 0=no
DEBUG=0             # 1=yes, 0=no
# ----------------------------------------

total=0; ok=0; fail=0

shopt -s nullglob
files=( "$DIR"/*."$EXT" )
if [ ${#files[@]} -eq 0 ]; then
  echo "No *.$EXT files found in: $DIR"
  exit 0
fi

for f in "${files[@]}"; do
  total=$((total+1))
  echo -e "\n[$total/${#files[@]}] ▶️  Processing: $f"

  # Fresh DDP comms each run (avoids port reuse issues)
  export MASTER_ADDR="127.0.0.1"
  export MASTER_PORT="$(shuf -i 29501-59999 -n 1)"

  # Pass run settings via env vars
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

  # Run ONE file per Python process (no loops inside Python)
  "$PYTHON" - <<'PY'
import os, sys, traceback
from pathlib import Path

root = os.environ["SR_ROOT"]
device = os.environ.get("SR_DEVICE","cuda")
gpu_ids = [int(x) for x in os.environ.get("SR_GPUS","").split(",") if x.strip()]
ws = (int(os.environ.get("SR_WS_H","128")), int(os.environ.get("SR_WS_W","128")))
factor = int(os.environ.get("SR_FACTOR","4"))
overlap = int(os.environ.get("SR_OVERLAP","8"))
elim = int(os.environ.get("SR_ELIM_BORDER","0"))
save_preview = os.environ.get("SR_SAVE_PREVIEW","0") == "1"
debug = os.environ.get("SR_DEBUG","0") == "1"

try:
    # Import here so each process is 100% isolated
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
except SystemExit as e:
    raise
except Exception:
    traceback.print_exc()
    sys.exit(1)
PY

  status=$?
  if [ $status -eq 0 ]; then
    echo "   ✅ Done: $f"
    ok=$((ok+1))
  else
    echo "   ❌ Failed: $f (exit $status)"
    fail=$((fail+1))
  fi
done

echo -e "\n📊 Summary → OK: $ok  FAIL: $fail  TOTAL: $total"
