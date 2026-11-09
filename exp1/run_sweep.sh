#!/bin/bash
# Sweep orchestrator. Runs one condition to completion, resumable, logged.
#
#   ./run_sweep.sh <condition> [seed] [tag]
#
# Examples:
#   ./run_sweep.sh linear              -> S1  seed 1337
#   ./run_sweep.sh graph               -> S2  seed 1337
#   ./run_sweep.sh linear 2026 rep     -> S3  seed 2026 (replicate / noise floor)
#   ./run_sweep.sh hybrid              -> S4  seed 1337
#
# Safe to re-run: harness_v2.py skips (instance, condition) pairs already in
# results.jsonl, so an interrupted sweep resumes where it stopped.

set -u

COND="${1:?usage: run_sweep.sh <linear|graph|hybrid> [seed] [tag]}"
SEED="${2:-1337}"
TAG="${3:-}"

BASE=/home/karthik/projects/kaggle/gemma4-agent-research
DS=$BASE/dataset
WORK=$BASE/exp1
MODEL=gemma4:31b
NUM_CTX="${NUM_CTX:-16384}"
# search_similar_code payload budget -- the experiment's independent variable.
# Payload scales with k x snippet_limit, so both move together:
#   lean: SNIPPET_LIMIT=200  SEARCH_K=5   -> ~1.6k chars/call  (~390 tokens)
#   fat : SNIPPET_LIMIT=1200 SEARCH_K=10  -> ~3.9k chars/call  (~965 tokens)
SNIPPET_LIMIT="${SNIPPET_LIMIT:-400}"
SEARCH_K="${SEARCH_K:-10}"
# WITH_SIGNATURES=1 makes get_code_neighbors return each neighbour's def/class
# line alongside its name -- the one-call variant tested in the paper. This
# departs from the official tool contract and is a proposed design change.
WITH_SIGNATURES="${WITH_SIGNATURES:-0}"
SIG_FLAG=""
[ "$WITH_SIGNATURES" = "1" ] && SIG_FLAG="--with-signatures"
# S4c: withhold run_bash_command for the opening N turns (0 disables).
SHELL_LOCK_LEN="${SHELL_LOCK_LEN:-0}"
SHELL_LOCK_CYCLE="${SHELL_LOCK_CYCLE:-0}"

RUN_ID="${COND}_ctx${NUM_CTX}_seed${SEED}${TAG:+_$TAG}"
OUT=$WORK/results/$RUN_ID
LOG=$WORK/logs/${RUN_ID}.log
mkdir -p "$OUT" "$WORK/logs"

echo "====================================================================="
echo " run_id      : $RUN_ID"
echo " condition   : $COND"
echo " model       : $MODEL   num_ctx=$NUM_CTX"
echo " sampling    : temp=1.0 top_p=0.95 top_k=64 seed=$SEED"
echo " search      : k=$SEARCH_K  snippet_limit=$SNIPPET_LIMIT chars/result"
echo " signatures  : $WITH_SIGNATURES (get_code_neighbors returns def lines)"
echo " shell lock  : len=$SHELL_LOCK_LEN cycle=$SHELL_LOCK_CYCLE (0 = no lock)"
echo " out         : $OUT"
echo " log         : $LOG"
echo " started     : $(date)  (unix $(date +%s))"
echo "====================================================================="

# --- GPU/memory telemetry, 10s cadence, coarse backstop to the per-turn log ---
TELEM=$OUT/telemetry.csv
if [ ! -f "$TELEM" ]; then
  echo "unix_ts,vram_gb,anon_gb,cached_gb,memfree_gb" > "$TELEM"
fi
(
  while true; do
    TS=$(date +%s)
    VRAM=$(rocm-smi --showmeminfo vram --csv 2>/dev/null \
           | awk -F, 'NR==2{printf "%.2f", $3/1024/1024/1024}')
    [ -z "$VRAM" ] && VRAM=$(rocm-smi --showmeminfo vram 2>/dev/null \
           | grep -oP 'VRAM Total Used Memory \(B\):\s*\K[0-9]+' \
           | head -1 | awk '{printf "%.2f", $1/1024/1024/1024}')
    read -r ANON CACHED FREE < <(awk '
      /^AnonPages:/{a=$2/1048576} /^Cached:/{c=$2/1048576} /^MemFree:/{f=$2/1048576}
      END{printf "%.2f %.2f %.2f", a, c, f}' /proc/meminfo)
    echo "$TS,$VRAM,$ANON,$CACHED,$FREE" >> "$TELEM"
    sleep 10
  done
) &
TELEM_PID=$!
trap 'kill $TELEM_PID 2>/dev/null' EXIT INT TERM

python3 -u "$WORK/harness_v2.py" \
  --dataset   "$DS" \
  --condition "$COND" \
  --out       "$OUT" \
  --model     "$MODEL" \
  --run-id    "$RUN_ID" \
  --num-ctx   "$NUM_CTX" \
  --temperature 1.0 --top-p 0.95 --top-k 64 \
  --seed      "$SEED" \
  --max-turns 20 \
  --snippet-limit "$SNIPPET_LIMIT" \
  --search-k "$SEARCH_K" \
  --shell-lock-len "$SHELL_LOCK_LEN" \
  --shell-lock-cycle "$SHELL_LOCK_CYCLE" \
  $SIG_FLAG \
  2>&1 | tee -a "$LOG"

STATUS=${PIPESTATUS[0]}
kill $TELEM_PID 2>/dev/null

# Guard: confirm the run actually used the settings it was asked for. A flag
# that is computed but never passed produces a silent duplicate of another
# condition, which is costly to discover after the fact.
python3 - "$OUT/results.jsonl" "$WITH_SIGNATURES" "$SNIPPET_LIMIT" "$SEARCH_K" "$SHELL_LOCK_LEN" <<'PY'
import json, sys
path, want_sig, want_snip, want_k = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
want_lock = sys.argv[5]
try:
    rows = [json.loads(l) for l in open(path) if l.strip()]
except OSError:
    sys.exit(0)
if not rows:
    sys.exit(0)
bad = []
got_sig = {str(r.get("with_signatures", 0)) for r in rows}
if got_sig != {want_sig}:
    bad.append(f"with_signatures: asked {want_sig}, logged {sorted(got_sig)}")
got_snip = {str(r.get("snippet_limit")) for r in rows}
if got_snip != {want_snip}:
    bad.append(f"snippet_limit: asked {want_snip}, logged {sorted(got_snip)}")
got_k = {str(r.get("search_k")) for r in rows}
if got_k != {want_k}:
    bad.append(f"search_k: asked {want_k}, logged {sorted(got_k)}")
got_lock = {str(r.get("shell_lock_len", 0)) for r in rows}
if got_lock != {want_lock}:
    bad.append(f"shell_lock_len: asked {want_lock}, logged {sorted(got_lock)}")
if bad:
    print()
    print("*** CONFIG MISMATCH -- this run did NOT use the requested settings ***")
    for b in bad:
        print("   ", b)
    print("    Do not use these results.")
else:
    print(f" config verified: with_signatures={want_sig} "
          f"snippet_limit={want_snip} search_k={want_k} "
          f"shell_lock_len={want_lock}")
PY

echo "====================================================================="
echo " finished    : $(date)"
echo " exit status : $STATUS"
echo " rows        : $(wc -l < "$OUT/results.jsonl" 2>/dev/null || echo 0)"
echo " answers     : $(ls "$OUT/answers" 2>/dev/null | wc -l)"
echo " traces      : $(ls "$OUT/traces" 2>/dev/null | wc -l)"
echo "====================================================================="
exit $STATUS
