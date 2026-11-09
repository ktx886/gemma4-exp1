#!/bin/bash
# S5 smoke: graph condition with get_code_neighbors returning signatures.
#
# The question is NOT "does it answer" -- S2a already answers 83/128. It is
# whether attaching signatures shifts the navigation:content call ratio, which
# sat at 11:89 in S2a and 9:91 in S2b. If signatures work, the agent should
# start using get_code_neighbors to decide where to go instead of paying for a
# search_similar_code call every hop.
#
#   ./smoke_s5.sh          20-turn budget, 6 tasks

set -u
TURNS="${1:-20}"
BASE=/home/karthik/projects/kaggle/gemma4-agent-research
WORK=$BASE/exp1
OUT=$WORK/results/smoke_s5

TASKS=(fastapi_14786 fastapi_14258 fastapi_14962 requests_7309 rich_4077 fastapi_15589)
ARGS=(); for t in "${TASKS[@]}"; do ARGS+=(--task-id "$t"); done

rm -rf "$OUT"; mkdir -p "$OUT"

echo "S5 smoke: ${#TASKS[@]} tasks, graph + signatures, max_turns=$TURNS"
echo

echo "--- payload pre-flight (CPU only) ---"
python3 - "$BASE/dataset" <<'PY'
import sys, json, io, os, statistics as st
sys.path.insert(0, "/home/karthik/projects/kaggle/gemma4-agent-research/exp1")
from harness_v2 import GraphTools
ds = sys.argv[1]
rows = [json.loads(l) for l in io.open(f"{ds}/tasks.jsonl", encoding="utf8")]
res = {}
for sig in (False, True):
    nb, sc = [], []
    for t in rows[:6]:
        pre = t["instance_id"].rsplit("_", 1)[0]
        gp = f"{ds}/graphs/{pre}_{t['base_commit']}.json"
        ep = f"{ds}/embeddings/{pre}_{t['base_commit']}.npz"
        if not os.path.exists(gp):
            continue
        G = GraphTools(gp, ep, snippet_limit=200, default_k=5, with_signatures=sig)
        probe = G.ids[len(G.ids) // 2].split(".")[-1]
        n, _ = G.get_code_neighbors(probe)
        s, _ = G.search_similar_code(probe)
        nb.append(len(n)); sc.append(len(s))
    res[sig] = (st.mean(nb), st.mean(sc))
    lab = "with signatures" if sig else "names only     "
    print(f"  {lab}  neighbours={st.mean(nb):6.0f} ch   search={st.mean(sc):6.0f} ch")
ok = True
if res[True][0] <= res[False][0] * 1.1:
    print("  !! FAIL: signatures did not increase the neighbours payload; "
          "the flag is probably not reaching GraphTools")
    ok = False
if res[True][0] > 3000:
    print(f"  !! FAIL: neighbours payload {res[True][0]:.0f} > 3000 ch")
    ok = False
sys.exit(0 if ok else 1)
PY
[ $? -ne 0 ] && { echo; echo "PRE-FLIGHT FAILED -- not running the model."; exit 1; }
echo "  pre-flight OK"
echo

python3 -u "$WORK/harness_v2.py" \
  --dataset "$BASE/dataset" --condition graph --out "$OUT" --run-id smoke_s5 \
  --model gemma4:31b --num-ctx 16384 \
  --temperature 1.0 --top-p 0.95 --top-k 64 --seed 1337 \
  --max-turns "$TURNS" --snippet-limit 200 --search-k 5 \
  --with-signatures \
  "${ARGS[@]}"

echo
echo "====================================================================="
python3 - "$OUT" <<'PY'
import json, sys, os, collections
out = sys.argv[1]
rows = [json.loads(l) for l in open(f"{out}/results.jsonl") if l.strip()]
print("stop reasons:", dict(collections.Counter(r["stop_reason"] for r in rows)))
print(f"non-empty answers: {sum(1 for r in rows if r['answer_chars'] > 0)}/{len(rows)}")
print(f"with_signatures logged: "
      f"{sorted({r.get('with_signatures') for r in rows})}")

mix, sz = collections.Counter(), collections.defaultdict(list)
for fn in os.listdir(f"{out}/traces"):
    for line in open(f"{out}/traces/{fn}"):
        r = json.loads(line)
        for c in r["calls"]:
            if c["name"] == "submit_answer":
                continue
            mix[c["name"]] += 1
            sz[c["name"]].append(c["result_chars"])
nav = mix["get_code_neighbors"] + mix["get_code_subgraph"]
con = mix["search_similar_code"]
tot = nav + con
print()
print(f"nav:content = {100*nav/tot:.0f}:{100*con/tot:.0f}   (S2a was 11:89)")
for k in sorted(sz):
    v = sz[k]
    print(f"  {k:22} n={len(v):4d} mean={sum(v)/len(v):6.0f} ch")
print()
if tot and 100 * nav / tot >= 25:
    print("SIGNAL: navigation share rose materially -- the one-call variant is")
    print("        changing behaviour. Worth the full sweep.")
else:
    print("NOTE: navigation share did not move much at n=6. Could be sample")
    print("      size; report before launching the full sweep.")
PY
echo "====================================================================="
