#!/bin/bash
# S4c smoke: hybrid with the shell withheld for the opening 5 turns.
#
# S4 (free choice, 128 tasks) gave 1,761 shell calls and 11 graph calls, and
# S4b (graph-first prompt) gave 0 graph calls in 78. The open question is no
# longer whether the model chooses the graph -- it does not -- but whether that
# choice is CORRECT. S4c withholds run_bash_command on turns 0-4 only, so the
# agent must open structurally and then has the shell for turns 5-19. Comparing
# S4c against S4 ON THE SAME TASKS measures what a forced graph start is worth.
#
#   ./smoke_s4c.sh        20-turn budget, 6 tasks

set -u
TURNS="${1:-20}"
LOCK_LEN="${LOCK_LEN:-5}"
LOCK_CYCLE="${LOCK_CYCLE:-0}"
BASE=/home/karthik/projects/kaggle/gemma4-agent-research
WORK=$BASE/exp1
OUT=$WORK/results/smoke_s4c

TASKS=(fastapi_14786 fastapi_14258 fastapi_14962 requests_7309 rich_4077 fastapi_15589)
ARGS=(); for t in "${TASKS[@]}"; do ARGS+=(--task-id "$t"); done

rm -rf "$OUT"; mkdir -p "$OUT"

echo "S4c smoke: ${#TASKS[@]} tasks, hybrid, shell withheld for turns 0-$((LOCK_LEN-1))"
echo

echo "--- lock schedule check (CPU only) ---"
python3 - "$LOCK_LEN" "$LOCK_CYCLE" "$TURNS" <<'PY'
import sys
sys.path.insert(0, "/home/karthik/projects/kaggle/gemma4-agent-research/exp1")
from harness_v2 import shell_locked, toolset
ln, cy, turns = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
sched = "".join("L" if shell_locked(t, ln, cy) else "." for t in range(turns))
print(f"  turns 0-{turns-1}: {sched}   (L = shell withheld)")
full = [t["function"]["name"] for t in toolset("hybrid")]
locked = [n for n in full if n != "run_bash_command"]
print(f"  unlocked tools: {full}")
print(f"  locked tools  : {locked}")
expected = (sum(1 for t in range(turns) if (t % cy) < ln) if cy > 0
            else min(ln, turns))
ok = sched.count("L") == expected and "run_bash_command" not in locked
print(f"  locked turns: {sched.count('L')} (expected {expected})")
print("  schedule valid:", ok)
sys.exit(0 if ok else 1)
PY
[ $? -ne 0 ] && { echo; echo "LOCK CHECK FAILED -- not running the model."; exit 1; }
echo "  lock check OK"
echo

python3 -u "$WORK/harness_v2.py" \
  --dataset "$BASE/dataset" --condition hybrid --out "$OUT" --run-id smoke_s4c \
  --model gemma4:31b --num-ctx 16384 \
  --temperature 1.0 --top-p 0.95 --top-k 64 --seed 1337 \
  --max-turns "$TURNS" --snippet-limit 200 --search-k 5 \
  --shell-lock-len "$LOCK_LEN" --shell-lock-cycle "$LOCK_CYCLE" \
  "${ARGS[@]}"

echo
echo "====================================================================="
python3 - "$OUT" <<'PY'
import json, sys, os, collections
out = sys.argv[1]
rows = [json.loads(l) for l in open(f"{out}/results.jsonl") if l.strip()]
print("stop reasons:", dict(collections.Counter(r["stop_reason"] for r in rows)))
print(f"non-empty answers: {sum(1 for r in rows if r['answer_chars'] > 0)}/{len(rows)}")
print(f"lock params logged: len={sorted({r.get('shell_lock_len') for r in rows})} "
      f"cycle={sorted({r.get('shell_lock_cycle') for r in rows})}")
print(f"mean locked turns per run: "
      f"{sum(r.get('locked_turns', 0) for r in rows)/len(rows):.1f}")

errs = [r for r in rows if r["stop_reason"] == "error"]
if errs:
    print()
    print(f"*** {len(errs)}/{len(rows)} runs ERRORED -- fix before the full run ***")
    print("   ", errs[0].get("error", "")[:160])
    sys.exit(1)

mix = collections.Counter()
for r in rows:
    mix.update(json.loads(r["tool_mix"]))
graph = {"get_code_neighbors", "search_similar_code", "get_code_subgraph"}
g = sum(mix[k] for k in graph); b = mix["run_bash_command"]; tot = g + b
if not tot:
    print()
    print("*** no tool calls recorded -- nothing to measure ***")
    sys.exit(1)
print()
print("tool mix:", {k: v for k, v in mix.most_common()})
print(f"  shell {b} ({100*b/tot:.1f}%)   graph {g} ({100*g/tot:.1f}%)"
      f"   (S4 free-choice was 98.8% / 1.2% on these tasks)")

# Did a shell call EXECUTE on a locked turn? An attempt that dispatch refused
# is not a violation -- it is the lock working, and is itself worth counting.
viol = attempts = 0
for fn in os.listdir(f"{out}/traces"):
    for line in open(f"{out}/traces/{fn}"):
        r = json.loads(line)
        if not r.get("shell_locked"):
            continue
        for c in r["calls"]:
            if c["name"] != "run_bash_command":
                continue
            if c.get("refused"):
                attempts += 1
            else:
                viol += 1
print(f"  shell calls EXECUTED on locked turns: {viol}  (must be 0)")
print(f"  shell calls refused on locked turns : {attempts}  "
      f"(the agent reaching for a withdrawn tool)")
print()
if viol:
    print("FAIL: the lock is not being enforced.")
elif tot and 100 * g / tot >= 10:
    print("OK: the lock forced structural navigation. Compare localization")
    print("    against S4 on these same 6 tasks before the full run.")
else:
    print("ODD: graph share still low despite the lock -- check whether the")
    print("     agent simply idled through locked turns.")
PY
echo "====================================================================="
