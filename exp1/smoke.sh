#!/bin/bash
# Smoke test on a MIXED-difficulty, multi-repo sample.
#
# The first two tasks in tasks.jsonl (fastapi_15588/15589) both failed in every
# single v1 run at every context size, so --limit 2 selects a pathological
# sample. These six span three repos and include tasks v1 solved, tasks v1
# failed, and one task that is graph-unreachable by construction.
#
#   ./smoke.sh            full 20-turn budget
#   ./smoke.sh 10         shorter budget

set -u
TURNS="${1:-20}"
SNIPPET_LIMIT="${SNIPPET_LIMIT:-400}"
SEARCH_K="${SEARCH_K:-10}"
WITH_SIGNATURES="${WITH_SIGNATURES:-0}"
SIG_FLAG=""
[ "$WITH_SIGNATURES" = "1" ] && SIG_FLAG="--with-signatures"
BASE=/home/karthik/projects/kaggle/gemma4-agent-research
WORK=$BASE/exp1
OUT=$WORK/results/smoke

# v1 outcomes at 16K for reference:
#   fastapi_14786  solved by BOTH      (easy, 1-2 tool calls)
#   fastapi_14258  solved by GRAPH only
#   fastapi_14962  solved by LINEAR only
#   requests_7309  solved by BOTH
#   rich_4077      solved by GRAPH only
#   fastapi_15589  solved by NEITHER   (hard; also graph-sparse)
TASKS=(fastapi_14786 fastapi_14258 fastapi_14962 requests_7309 rich_4077 fastapi_15589)

ARGS=()
for t in "${TASKS[@]}"; do ARGS+=(--task-id "$t"); done

rm -rf "$OUT"
mkdir -p "$OUT"

echo "smoke: ${#TASKS[@]} tasks x 3 conditions, max_turns=$TURNS"
echo "snippet_limit=$SNIPPET_LIMIT chars/result   search_k=$SEARCH_K   signatures=$WITH_SIGNATURES"
echo "tasks: ${TASKS[*]}"
echo

# --- pre-flight: verify tool payload sizes WITHOUT burning GPU time ---------
echo "--- payload pre-flight (CPU only) ---"
python3 - "$BASE/dataset" "$SNIPPET_LIMIT" "$SEARCH_K" "$WITH_SIGNATURES" <<'PY'
import sys, json, io, os, statistics as st
sys.path.insert(0, "/home/karthik/projects/kaggle/gemma4-agent-research/exp1")
from harness_v2 import GraphTools
ds, lim, kk = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
sig = len(sys.argv) > 4 and sys.argv[4] == "1"
rows = [json.loads(l) for l in io.open(f"{ds}/tasks.jsonl", encoding="utf8")]
sizes = {"search_similar_code": [], "get_code_neighbors": []}
for t in rows[:6]:
    pre = t["instance_id"].rsplit("_", 1)[0]
    gp = f"{ds}/graphs/{pre}_{t['base_commit']}.json"
    ep = f"{ds}/embeddings/{pre}_{t['base_commit']}.npz"
    if not os.path.exists(gp):
        continue
    G = GraphTools(gp, ep, snippet_limit=lim, default_k=kk, with_signatures=sig)
    probe = G.ids[len(G.ids) // 2].split(".")[-1]
    s, _ = G.search_similar_code(probe)
    n, _ = G.get_code_neighbors(probe)
    sizes["search_similar_code"].append(len(s))
    sizes["get_code_neighbors"].append(len(n))
ok = True
for k, v in sizes.items():
    if not v:
        continue
    m = st.mean(v)
    print(f"  {k:22} mean={m:6.0f} max={max(v):6d} chars  (~{m/4:.0f} tokens)")
    if k == "search_similar_code" and m > 4200:
        print(f"  !! FAIL: {k} mean {m:.0f} > 4200; would saturate a 16K window")
        ok = False
    lim_n = 3000 if sig else 1500
    if k == "get_code_neighbors" and m > lim_n:
        print(f"  !! FAIL: {k} mean {m:.0f} > {lim_n}")
        ok = False
sys.exit(0 if ok else 1)
PY
if [ $? -ne 0 ]; then
  echo
  echo "PRE-FLIGHT FAILED -- payloads out of range. Not running the model."
  exit 1
fi
echo "  pre-flight OK"
echo

for c in linear graph hybrid; do
  echo "--- $c ---"
  python3 -u "$WORK/harness_v2.py" \
    --dataset "$BASE/dataset" \
    --condition "$c" \
    --out "$OUT" \
    --run-id "smoke_$c" \
    --model gemma4:31b \
    --num-ctx 16384 \
    --temperature 1.0 --top-p 0.95 --top-k 64 --seed 1337 \
    --max-turns "$TURNS" \
    --snippet-limit "$SNIPPET_LIMIT" \
    --search-k "$SEARCH_K" \
    $SIG_FLAG \
    "${ARGS[@]}"
  echo
done

echo "====================================================================="
echo "stop_reason distribution:"
python3 - "$OUT/results.jsonl" <<'PY'
import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
by = collections.defaultdict(collections.Counter)
for r in rows:
    by[r["condition"]][r["stop_reason"]] += 1
for c in ("linear", "graph", "hybrid"):
    if c in by:
        print(f"  {c:8} {dict(by[c])}")
ans = [r for r in rows if r["answer_chars"] > 0]
print(f"\n  runs with a non-empty answer: {len(ans)}/{len(rows)}")
if ans:
    print(f"  median answer chars: "
          f"{sorted(r['answer_chars'] for r in ans)[len(ans)//2]}")
print(f"  distinct stop reasons: "
      f"{sorted({r['stop_reason'] for r in rows})}")
mix = collections.Counter()
for r in rows:
    if r["condition"] == "hybrid":
        mix.update(json.loads(r["tool_mix"]))
if mix:
    print(f"\n  hybrid tool mix: {dict(mix)}")
PY
echo "====================================================================="
