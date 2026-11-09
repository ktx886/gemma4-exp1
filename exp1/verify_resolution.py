#!/usr/bin/env python3
"""Verify symbol resolution in the graph tools. CPU-only, no LLM.

For every task, take the gold-patch symbols -- the exact names a competent
agent would search for -- and check that search_similar_code and
get_code_neighbors actually resolve them. A high miss rate means graph mode is
handicapped by a lookup defect rather than by the modality, which would
invalidate the linear-vs-graph comparison.

Usage:
    python3 verify_resolution.py --dataset ../dataset
    python3 verify_resolution.py --dataset ../dataset --verbose
"""

import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness_v2 import GraphTools  # noqa: E402


def gold_symbols(patch):
    s = set(re.findall(r"^@@.*@@\s*(?:async\s+)?(?:def|class)\s+(\w+)", patch, re.M))
    s |= set(re.findall(r"^[+-]\s*(?:async\s+)?(?:def|class)\s+(\w+)", patch, re.M))
    return {x for x in s if len(x) > 2 and not x.startswith("__")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    with open(os.path.join(args.dataset, "tasks.jsonl"), encoding="utf8") as f:
        tasks = [json.loads(l) for l in f if l.strip()]
    if args.limit:
        tasks = tasks[:args.limit]

    tot = hit_search = hit_neigh = 0
    by_repo = collections.defaultdict(lambda: [0, 0, 0])
    misses = []

    for n, t in enumerate(tasks, 1):
        inst = t["instance_id"]
        pre = inst.rsplit("_", 1)[0]
        gp = os.path.join(args.dataset, "graphs", f"{pre}_{t['base_commit']}.json")
        ep = os.path.join(args.dataset, "embeddings", f"{pre}_{t['base_commit']}.npz")
        if not os.path.exists(gp):
            continue
        G = GraphTools(gp, ep)
        syms = sorted(gold_symbols(t["patch"]))
        if not syms:
            continue
        repo = t["repo"].split("/")[-1]

        s_ok = n_ok = False
        s_detail = n_detail = ""
        for s in syms:
            txt, cnt = G.search_similar_code(s, k=5)
            if cnt > 0:
                s_ok = True
                s_detail = s
                break
        for s in syms:
            txt, cnt = G.get_code_neighbors(s, 10)
            if cnt > 0:
                n_ok = True
                n_detail = s
                break

        tot += 1
        hit_search += s_ok
        hit_neigh += n_ok
        by_repo[repo][0] += 1
        by_repo[repo][1] += s_ok
        by_repo[repo][2] += n_ok
        if not s_ok:
            misses.append((inst, repo, syms[:4]))
        if args.verbose:
            print(f"{inst:18} {repo:9} search={'OK ' if s_ok else 'MISS'}"
                  f"({s_detail or syms[0]:28.28}) "
                  f"neighbors={'OK ' if n_ok else 'MISS'}({n_detail or syms[0]:28.28})")
        elif n % 20 == 0:
            print(f"  {n}/{len(tasks)}", flush=True)

    print("\n" + "=" * 70)
    print("SYMBOL RESOLUTION ON GOLD-PATCH TARGETS")
    print("=" * 70)
    print(f"tasks with gold symbols: {tot}")
    print(f"  search_similar_code resolves >=1 gold symbol: "
          f"{hit_search} ({100*hit_search/max(tot,1):.1f}%)")
    print(f"  get_code_neighbors  resolves >=1 gold symbol: "
          f"{hit_neigh} ({100*hit_neigh/max(tot,1):.1f}%)")
    print()
    print(f"{'repo':10}{'n':>5}{'search':>10}{'neighbors':>12}")
    for r, v in sorted(by_repo.items()):
        print(f"{r:10}{v[0]:5d}{100*v[1]/v[0]:9.1f}%{100*v[2]/v[0]:11.1f}%")

    if misses:
        print(f"\n{len(misses)} tasks where NO gold symbol resolves in search:")
        for i, r, s in misses[:15]:
            print(f"  {i:18} {r:9} {s}")

    print()
    if tot and 100 * hit_search / tot >= 85:
        print("PASS: resolution is not the bottleneck; graph mode is fairly tested.")
    else:
        print("FAIL: resolution misses too often. Fix before running the sweep --")
        print("      graph results would measure the lookup defect, not the modality.")


if __name__ == "__main__":
    main()
