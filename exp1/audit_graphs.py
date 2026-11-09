#!/usr/bin/env python3
"""Audit the competition's code graphs and embeddings. CPU-only: no GPU, no LLM.

Measurements
------------
A1  graph topology    : nodes, edges, degree, isolated %, split source vs test/docs
A2  namespace mix     : what fraction of nodes are library source vs tests/docs/scripts
A3  oracle reach      : is the gold-patch symbol present as a node? does it have edges?
A4  retrieval noise   : run the official symbol-similarity search for the gold symbol;
                        what fraction of the top-k hits are test/doc nodes?
A5  hop distance      : shortest path from a plausible entry node to the gold target
A6  snapshot fairness : is the gold file actually present in the .tgz the agent gets?

Outputs (to --out):
    graph_metrics.csv       one row per distinct graph file
    task_reachability.csv   one row per task (A3, A5, A6)
    pollution.csv           one row per task (A4)
    audit_summary.md        human-readable rollup

Usage:
    python3 audit_graphs.py --dataset /path/to/dataset --out audit/
"""

import argparse
import collections
import csv
import json
import os
import re
import sys
import tarfile

import numpy as np

# Top-level namespaces that are NOT library source code. Used to separate
# "the code the agent must fix" from "test and documentation scaffolding".
NON_SOURCE_NS = {
    "tests", "test", "testing", "docs", "docs_src", "doc",
    "scripts", "benchmarks", "benchmark", "examples", "example", "tools",
}


# ---------------------------------------------------------------------------
# gold-patch parsing
# ---------------------------------------------------------------------------

def gold_files(patch):
    """Files touched by the gold patch (unified diff headers)."""
    fs = set(re.findall(r"^\+\+\+ b/(.+?)\s*$", patch, re.M))
    fs |= set(re.findall(r"^--- a/(.+?)\s*$", patch, re.M))
    return {f for f in fs if f != "/dev/null"}


def gold_symbols(patch):
    """Function/class names named in @@ hunk headers."""
    return set(re.findall(r"^@@.*@@\s*(?:async\s+)?(?:def|class)\s+(\w+)", patch, re.M))


def module_candidates(filepath):
    """Dotted-module candidates for a file path, tolerant of src/ layouts.

    'src/requests/models.py' -> ['src.requests.models', 'requests.models', 'models']
    """
    if not filepath.endswith(".py"):
        return []
    stem = filepath[:-3]
    if stem.endswith("/__init__"):
        stem = stem[:-9]
    parts = stem.split("/")
    return [".".join(parts[i:]) for i in range(len(parts))]


def is_source_node(node_id):
    return node_id.split(".")[0] not in NON_SOURCE_NS


# ---------------------------------------------------------------------------
# graph loading
# ---------------------------------------------------------------------------

class Graph:
    def __init__(self, path):
        with open(path) as f:
            g = json.load(f)
        self.nodes = {n["id"]: n for n in g["nodes"]}
        self.ids = list(self.nodes)
        self.edges = g["edges"]
        self.deg = collections.Counter()
        self.adj = collections.defaultdict(set)
        for e in self.edges:
            s, t = e["source"], e["target"]
            self.deg[s] += 1
            self.deg[t] += 1
            self.adj[s].add(t)
            self.adj[t].add(s)

    def resolve(self, name):
        """Official 4-tier symbol resolution: exact, suffix, case-insensitive, substring."""
        if name in self.nodes:
            return [name]
        hits = [i for i in self.ids if i.endswith("." + name)]
        if hits:
            return hits
        low = name.lower()
        hits = [i for i in self.ids if i.lower() == low]
        if hits:
            return hits
        return [i for i in self.ids if low in i.lower()][:50]

    def hops(self, src_ids, dst_ids, cap=6):
        """Shortest undirected hop count between any src and any dst, or -1."""
        dst = set(dst_ids)
        seen = set(src_ids)
        frontier = set(src_ids)
        if seen & dst:
            return 0
        for d in range(1, cap + 1):
            nxt = set()
            for n in frontier:
                nxt |= self.adj[n]
            nxt -= seen
            if not nxt:
                return -1
            if nxt & dst:
                return d
            seen |= nxt
            frontier = nxt
        return -1


# ---------------------------------------------------------------------------
# A4: official-semantics embedding search
# ---------------------------------------------------------------------------

class Embeddings:
    """Mirrors swegemma search_similar_code: resolve query to a node, then
    return that node's nearest neighbours by cosine similarity."""

    def __init__(self, path):
        z = np.load(path, allow_pickle=True)
        self.keys = list(z.keys())
        mat = np.stack([z[k] for k in self.keys]).astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.mat = mat / norms
        self.index = {k: i for i, k in enumerate(self.keys)}

    def resolve(self, name):
        if name in self.index:
            return name
        for k in self.keys:
            if k.endswith("." + name):
                return k
        low = name.lower()
        for k in self.keys:
            if k.lower() == low:
                return k
        for k in self.keys:
            if low in k.lower():
                return k
        return None

    def neighbors(self, name, k=10):
        key = self.resolve(name)
        if key is None:
            return []
        i = self.index[key]
        sims = self.mat @ self.mat[i]
        sims[i] = -2.0
        top = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
        top = top[np.argsort(-sims[top])]
        return [(self.keys[j], float(sims[j])) for j in top]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="audit")
    ap.add_argument("--topk", type=int, default=5, help="top-k for A4 pollution")
    ap.add_argument("--skip-snapshots", action="store_true",
                    help="skip A6 (slow: opens every .tgz)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ds = args.dataset

    with open(os.path.join(ds, "tasks.jsonl"), encoding="utf8") as f:
        tasks = [json.loads(l) for l in f if l.strip()]
    print(f"tasks: {len(tasks)}", flush=True)

    # ---------------- A1 / A2 : per-graph topology ----------------
    graph_rows = []
    seen = {}
    for t in tasks:
        pre = t["instance_id"].rsplit("_", 1)[0]
        gp = os.path.join(ds, "graphs", f"{pre}_{t['base_commit']}.json")
        if gp in seen or not os.path.exists(gp):
            continue
        seen[gp] = True
        g = Graph(gp)
        src = [i for i in g.ids if is_source_node(i)]
        nonsrc = [i for i in g.ids if not is_source_node(i)]
        iso = sum(1 for i in g.ids if g.deg[i] == 0)
        iso_src = sum(1 for i in src if g.deg[i] == 0)
        degs = sorted(g.deg[i] for i in src) or [0]
        graph_rows.append({
            "graph": os.path.basename(gp),
            "repo": t["repo"].split("/")[-1],
            "nodes": len(g.ids),
            "edges": len(g.edges),
            "src_nodes": len(src),
            "nonsrc_nodes": len(nonsrc),
            "pct_nonsrc": round(100 * len(nonsrc) / max(len(g.ids), 1), 2),
            "isolated_all": iso,
            "pct_isolated_all": round(100 * iso / max(len(g.ids), 1), 2),
            "isolated_src": iso_src,
            "pct_isolated_src": round(100 * iso_src / max(len(src), 1), 2),
            "mean_deg_src": round(sum(degs) / len(degs), 3),
            "median_deg_src": degs[len(degs) // 2],
        })
        print(f"  A1 {os.path.basename(gp)[:42]:44} "
              f"iso_src={graph_rows[-1]['pct_isolated_src']:5.1f}%", flush=True)

    _write_csv(os.path.join(args.out, "graph_metrics.csv"), graph_rows)

    # ---------------- A3 / A5 / A6 : per-task reachability ----------------
    reach_rows, poll_rows = [], []
    emb_cache = {}
    for n, t in enumerate(tasks, 1):
        inst = t["instance_id"]
        pre = inst.rsplit("_", 1)[0]
        gp = os.path.join(ds, "graphs", f"{pre}_{t['base_commit']}.json")
        ep = os.path.join(ds, "embeddings", f"{pre}_{t['base_commit']}.npz")
        if not os.path.exists(gp):
            continue
        g = Graph(gp)
        gfiles = gold_files(t["patch"])
        gpy = {f for f in gfiles if f.endswith(".py")}
        gsyms = gold_symbols(t["patch"])

        # A3: file reachable as a node prefix
        file_nodes = []
        for f in gpy:
            for m in module_candidates(f):
                hits = [i for i in g.ids if i == m or i.startswith(m + ".")]
                if hits:
                    file_nodes = hits
                    break
            if file_nodes:
                break
        sym_nodes = [i for i in g.ids
                     if any(i == s or i.endswith("." + s) for s in gsyms)]
        sym_connected = [i for i in sym_nodes if g.deg[i] > 0]
        target_nodes = sym_connected or [i for i in file_nodes if g.deg[i] > 0]

        # A5: hops from the repo's highest-degree source node to the target
        src_by_deg = sorted((i for i in g.ids if is_source_node(i)),
                            key=lambda i: -g.deg[i])
        entry = src_by_deg[:3]
        hops = g.hops(entry, target_nodes) if (entry and target_nodes) else -1

        # A6: gold file present in the snapshot
        snap_ok = ""
        if not args.skip_snapshots:
            sp = os.path.join(ds, "snapshots", f"{inst}.tgz")
            if os.path.exists(sp):
                try:
                    with tarfile.open(sp) as tf:
                        names = {x.lstrip("./") for x in tf.getnames()}
                    roots = {x.split("/")[0] for x in names if "/" in x}
                    snap_ok = int(any(
                        f in names or any(f"{r}/{f}" in names for r in roots)
                        for f in gpy))
                except Exception as e:                      # noqa: BLE001
                    snap_ok = f"err:{type(e).__name__}"

        reach_rows.append({
            "instance_id": inst,
            "repo": t["repo"].split("/")[-1],
            "n_gold_files": len(gpy),
            "n_gold_symbols": len(gsyms),
            "file_reachable": int(bool(file_nodes)),
            "symbol_present": int(bool(sym_nodes)),
            "symbol_connected": int(bool(sym_connected)),
            "hops_to_target": hops,
            "gold_in_snapshot": snap_ok,
            "problem_len": len(t["problem_statement"]),
            "patch_len": len(t["patch"]),
        })

        # ---------------- A4 : retrieval pollution ----------------
        if os.path.exists(ep) and gsyms:
            if ep not in emb_cache:
                if len(emb_cache) > 3:
                    emb_cache.clear()
                emb_cache[ep] = Embeddings(ep)
            E = emb_cache[ep]
            probe = sorted(gsyms)[0]
            hits = E.neighbors(probe, k=args.topk)
            if hits:
                nonsrc = sum(1 for h, _ in hits if not is_source_node(h))
                poll_rows.append({
                    "instance_id": inst,
                    "repo": t["repo"].split("/")[-1],
                    "probe_symbol": probe,
                    "resolved": int(E.resolve(probe) is not None),
                    "k": len(hits),
                    "n_nonsource_hits": nonsrc,
                    "pct_nonsource": round(100 * nonsrc / len(hits), 1),
                    "top1": hits[0][0],
                    "top1_sim": round(hits[0][1], 4),
                    "top1_is_source": int(is_source_node(hits[0][0])),
                })
        if n % 10 == 0:
            print(f"  A3/A4/A5 {n}/{len(tasks)}", flush=True)

    _write_csv(os.path.join(args.out, "task_reachability.csv"), reach_rows)
    _write_csv(os.path.join(args.out, "pollution.csv"), poll_rows)

    _summary(args.out, graph_rows, reach_rows, poll_rows)
    print(f"\nwrote {args.out}/graph_metrics.csv, task_reachability.csv, "
          f"pollution.csv, audit_summary.md")


def _write_csv(path, rows):
    if not rows:
        open(path, "w").close()
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _summary(out, graph_rows, reach_rows, poll_rows):
    L = []
    A = L.append
    A("# Graph / embedding audit\n")

    A("## A1-A2 Topology and composition by repo\n")
    A("| repo | graphs | nodes | %non-source | %isolated (all) | %isolated (source) | mean deg (source) |")
    A("|---|---|---|---|---|---|---|")
    by = collections.defaultdict(list)
    for r in graph_rows:
        by[r["repo"]].append(r)
    for repo, rs in sorted(by.items()):
        tn = sum(r["nodes"] for r in rs)
        ts = sum(r["src_nodes"] for r in rs)
        A(f"| {repo} | {len(rs)} | {tn} | "
          f"{100*sum(r['nonsrc_nodes'] for r in rs)/max(tn,1):.1f}% | "
          f"{100*sum(r['isolated_all'] for r in rs)/max(tn,1):.1f}% | "
          f"**{100*sum(r['isolated_src'] for r in rs)/max(ts,1):.1f}%** | "
          f"{sum(r['mean_deg_src'] for r in rs)/len(rs):.2f} |")

    A("\n## A3 Oracle reachability (ceiling for graph navigation)\n")
    A("| repo | n | file reachable | symbol present | symbol connected |")
    A("|---|---|---|---|---|")
    by = collections.defaultdict(list)
    for r in reach_rows:
        by[r["repo"]].append(r)
    for repo, rs in sorted(by.items()):
        n = len(rs)
        A(f"| {repo} | {n} | {100*sum(r['file_reachable'] for r in rs)/n:.1f}% | "
          f"{100*sum(r['symbol_present'] for r in rs)/n:.1f}% | "
          f"**{100*sum(r['symbol_connected'] for r in rs)/n:.1f}%** |")
    n = len(reach_rows)
    if n:
        A(f"\n**Overall: symbol connected (traversable) = "
          f"{100*sum(r['symbol_connected'] for r in reach_rows)/n:.1f}%**")

    A("\n## A5 Hop distance from hub entry node to gold target\n")
    hs = [r["hops_to_target"] for r in reach_rows if r["hops_to_target"] >= 0]
    unreach = sum(1 for r in reach_rows if r["hops_to_target"] < 0)
    if hs:
        A(f"- reachable within 6 hops: {len(hs)}/{n} ({100*len(hs)/n:.1f}%)")
        A(f"- unreachable (>6 hops or disconnected): {unreach} ({100*unreach/n:.1f}%)")
        A(f"- hop histogram: {dict(sorted(collections.Counter(hs).items()))}")

    A("\n## A6 Snapshot fairness (linear condition)\n")
    ok = sum(1 for r in reach_rows if r["gold_in_snapshot"] == 1)
    bad = [r["instance_id"] for r in reach_rows if r["gold_in_snapshot"] == 0]
    A(f"- gold file present in snapshot: {ok}/{n} ({100*ok/max(n,1):.1f}%)")
    if bad:
        A(f"- MISSING (exclude these tasks): {bad}")

    A("\n## A4 Retrieval pollution (official symbol-similarity top-k)\n")
    if poll_rows:
        A("| repo | n | mean %non-source in top-k | top-1 is source |")
        A("|---|---|---|---|")
        by = collections.defaultdict(list)
        for r in poll_rows:
            by[r["repo"]].append(r)
        for repo, rs in sorted(by.items()):
            A(f"| {repo} | {len(rs)} | "
              f"**{sum(r['pct_nonsource'] for r in rs)/len(rs):.1f}%** | "
              f"{100*sum(r['top1_is_source'] for r in rs)/len(rs):.1f}% |")
        A(f"\nOverall mean non-source share of top-k: "
          f"**{sum(r['pct_nonsource'] for r in poll_rows)/len(poll_rows):.1f}%**")
        A("\n*High non-source share means the embedding search hands the agent "
          "test/doc nodes instead of library code -- a precision problem with a "
          "one-line fix (namespace filtering).*")
    else:
        A("_no pollution rows produced_")

    with open(os.path.join(out, "audit_summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
