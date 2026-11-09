#!/usr/bin/env python3
"""Analysis for the three claims.

C1  independence : do linear and graph solve near-disjoint task sets, and is the
                   overlap at chance? Compared against the linear-vs-linear
                   replicate, which fixes the run-to-run noise floor.
C2  mechanism    : does per-repo win rate track graph composition (audit join)?
C3  support      : stop-reason mix; payload asymmetry on matched-turn pairs.

Usage:
    python3 analyze.py --scored scored.csv [--audit audit/] [--out analysis.md]
"""

import argparse
import collections
import csv
import json
import math
import os
import statistics as st

METRIC = "loc_any"          # primary success metric; switch to loc_strict for the
                            # conservative variant


def binom_two_sided(b, c):
    """Exact two-sided sign / McNemar test on discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    lo = min(b, c)
    p = sum(math.comb(n, i) for i in range(lo + 1)) * (0.5 ** n) * 2
    return min(1.0, p)


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("turns", "tool_calls", "prompt_tokens", "completion_tokens",
                  "answer_chars", "zero_result_calls", "loc_file", "loc_symbol",
                  "loc_strict", "loc_any", "thinking_tokens_est"):
            if k in r and r[k] != "":
                try:
                    r[k] = int(float(r[k]))
                except ValueError:
                    r[k] = 0
        for k in ("elapsed_s", "setup_s", "agent_s"):
            if k in r and r[k] != "":
                try:
                    r[k] = float(r[k])
                except ValueError:
                    r[k] = 0.0
    return rows


def index(rows, run_id):
    return {r["instance_id"]: r for r in rows if r.get("run_id") == run_id}


def pair_table(A, B, nameA, nameB, out):
    ids = sorted(set(A) & set(B))
    both = sum(1 for i in ids if A[i][METRIC] and B[i][METRIC])
    onlyA = sum(1 for i in ids if A[i][METRIC] and not B[i][METRIC])
    onlyB = sum(1 for i in ids if not A[i][METRIC] and B[i][METRIC])
    neither = len(ids) - both - onlyA - onlyB
    rA = (both + onlyA) / len(ids) if ids else 0
    rB = (both + onlyB) / len(ids) if ids else 0
    exp = rA * rB * len(ids)
    union = both + onlyA + onlyB
    jac = both / union if union else 0.0
    out.append(f"\n### {nameA} vs {nameB}  (n={len(ids)} paired)\n")
    out.append(f"| | {nameB} solved | {nameB} failed |")
    out.append("|---|---|---|")
    out.append(f"| **{nameA} solved** | {both} | {onlyA} |")
    out.append(f"| **{nameA} failed** | {onlyB} | {neither} |")
    out.append("")
    out.append(f"- success: {nameA} {100*rA:.1f}%, {nameB} {100*rB:.1f}%")
    out.append(f"- overlap observed **{both}**, expected under independence "
               f"**{exp:.1f}** -> ratio **{both/exp if exp else 0:.2f}x**")
    out.append(f"- Jaccard overlap: **{100*jac:.1f}%**")
    out.append(f"- union (oracle router): **{union}/{len(ids)} = {100*union/len(ids):.1f}%** "
               f"vs best single {100*max(rA,rB):.1f}% "
               f"(**+{100*(union/len(ids)-max(rA,rB)):.1f} pp**)")
    out.append(f"- McNemar exact p = **{binom_two_sided(onlyA, onlyB):.4f}** "
               f"(tests whether either is better overall)")
    return {"n": len(ids), "both": both, "onlyA": onlyA, "onlyB": onlyB,
            "union": union, "exp": exp, "jaccard": jac,
            "rA": rA, "rB": rB}


def main():
    global METRIC

    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--audit", default="")
    ap.add_argument("--out", default="analysis.md")
    ap.add_argument("--metric", default=METRIC,
                    choices=["loc_any", "loc_file", "loc_symbol", "loc_strict"])
    args = ap.parse_args()

    METRIC = args.metric

    rows = load(args.scored)
    runs = sorted({r.get("run_id", "") for r in rows})
    L = []
    A = L.append
    A(f"# Analysis (success metric: `{METRIC}`)\n")
    A(f"runs present: {runs}\n")

    # ---------------- per-run summary ----------------
    A("## Per-run summary\n")
    A("| run_id | cond | n | success | answered | max_turns | empty | err | "
      "turns | calls | zero-result | p_tok | c_tok |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    byrun = collections.defaultdict(list)
    for r in rows:
        byrun[r.get("run_id", "")].append(r)
    for rid in runs:
        rs = byrun[rid]
        n = len(rs)
        sr = collections.Counter(r["stop_reason"] for r in rs)
        A(f"| {rid} | {rs[0]['condition']} | {n} | "
          f"**{100*sum(r[METRIC] for r in rs)/n:.1f}%** | "
          f"{100*sr['final_answer']/n:.0f}% | {100*sr['max_turns']/n:.0f}% | "
          f"{100*sr['empty_response']/n:.0f}% | {sr['error']} | "
          f"{st.mean(r['turns'] for r in rs):.1f} | "
          f"{st.mean(r['tool_calls'] for r in rs):.1f} | "
          f"{st.mean(r['zero_result_calls'] for r in rs):.1f} | "
          f"{st.mean(r['prompt_tokens'] for r in rs):.0f} | "
          f"{st.mean(r['completion_tokens'] for r in rs):.0f} |")

    # ---------------- C1 ----------------
    A("\n## C1  Independence of failure modes\n")
    lin = [r for r in runs if r.startswith("linear") and "_rep" not in r]
    rep = [r for r in runs if r.startswith("linear") and "_rep" in r]
    gra = [r for r in runs if r.startswith("graph")]
    hyb = [r for r in runs if r.startswith("hybrid")]

    stats = {}
    if lin and gra:
        stats["lin_graph"] = pair_table(index(rows, lin[0]), index(rows, gra[0]),
                                        "linear", "graph", L)
    if lin and rep:
        A("\n**Noise floor** - same condition, different seed. If this overlap is "
          "high while linear-vs-graph is at chance, the disjointness is a property "
          "of the tooling, not of sampling variance.\n")
        stats["lin_rep"] = pair_table(index(rows, lin[0]), index(rows, rep[0]),
                                      "linear(1337)", "linear(2026)", L)
    if lin and gra and hyb:
        stats["lin_hyb"] = pair_table(index(rows, lin[0]), index(rows, hyb[0]),
                                      "linear", "hybrid", L)
        stats["gra_hyb"] = pair_table(index(rows, gra[0]), index(rows, hyb[0]),
                                      "graph", "hybrid", L)

    if "lin_graph" in stats and "lin_rep" in stats:
        a, b = stats["lin_graph"], stats["lin_rep"]
        A("\n### Verdict on C1\n")
        A(f"- linear vs graph Jaccard: **{100*a['jaccard']:.1f}%** "
          f"(overlap {a['both']/a['exp'] if a['exp'] else 0:.2f}x chance)")
        A(f"- linear vs linear Jaccard: **{100*b['jaccard']:.1f}%** "
          f"(overlap {b['both']/b['exp'] if b['exp'] else 0:.2f}x chance)")
        if b["jaccard"] > a["jaccard"] * 1.5:
            A("\n**C1 SUPPORTED.** Same-condition replication overlaps far more than "
              "cross-condition, so the disjointness is attributable to retrieval "
              "modality rather than sampling noise.")
        else:
            A("\n**C1 NOT SUPPORTED.** Same-condition replication overlaps about as "
              "little as cross-condition: the apparent disjointness is largely "
              "run-to-run variance. Report this honestly and pivot the paper to the "
              "audit + C3.")
    if "lin_hyb" in stats and "lin_graph" in stats:
        h = stats["lin_hyb"]["rB"]
        orc = stats["lin_graph"]["union"] / stats["lin_graph"]["n"]
        best = max(stats["lin_graph"]["rA"], stats["lin_graph"]["rB"])
        A(f"\n### Oracle vs realized routing\n")
        A(f"- best single modality : **{100*best:.1f}%**")
        A(f"- hybrid (model routes): **{100*h:.1f}%**")
        A(f"- oracle (perfect route): **{100*orc:.1f}%**")
        gap = orc - h
        A(f"- **realized-vs-oracle gap: {100*gap:.1f} pp** "
          f"({100*(h-best):+.1f} pp vs best single)")

    # ---------------- C2 ----------------
    A("\n## C2  Per-repo breakdown (mechanism)\n")
    A("| repo | " + " | ".join(runs) + " |")
    A("|---" * (len(runs) + 1) + "|")
    repos = sorted({r["repo"] for r in rows})
    for repo in repos:
        cells = []
        for rid in runs:
            rs = [r for r in byrun[rid] if r["repo"] == repo]
            cells.append(f"{100*sum(r[METRIC] for r in rs)/len(rs):.0f}% (n={len(rs)})"
                         if rs else "-")
        A(f"| {repo} | " + " | ".join(cells) + " |")

    if args.audit:
        gm = os.path.join(args.audit, "graph_metrics.csv")
        pol = os.path.join(args.audit, "pollution.csv")
        if os.path.exists(gm):
            with open(gm) as f:
                g = list(csv.DictReader(f))
            agg = collections.defaultdict(list)
            for r in g:
                agg[r["repo"]].append(float(r["pct_isolated_src"]))
            A("\n### Join with graph composition\n")
            A("| repo | %isolated (source) | mean %non-source in top-k |")
            A("|---|---|---|")
            pm = {}
            if os.path.exists(pol):
                with open(pol) as f:
                    pr = list(csv.DictReader(f))
                pa = collections.defaultdict(list)
                for r in pr:
                    pa[r["repo"]].append(float(r["pct_nonsource"]))
                pm = {k: st.mean(v) for k, v in pa.items()}
            for repo in sorted(agg):
                A(f"| {repo} | {st.mean(agg[repo]):.1f}% | "
                  f"{pm.get(repo, float('nan')):.1f}% |")

    # ---------------- C3 ----------------
    A("\n## C3  Stop-reason mix and payload asymmetry\n")
    A("| run_id | final_answer | max_turns | empty | error |")
    A("|---|---|---|---|---|")
    for rid in runs:
        rs = byrun[rid]
        n = len(rs)
        c = collections.Counter(r["stop_reason"] for r in rs)
        A(f"| {rid} | {100*c['final_answer']/n:.0f}% | {100*c['max_turns']/n:.0f}% | "
          f"{100*c['empty_response']/n:.0f}% | {100*c['error']/n:.0f}% |")

    if lin and gra:
        Lx, Gx = index(rows, lin[0]), index(rows, gra[0])
        ids = [i for i in set(Lx) & set(Gx)
               if Lx[i]["turns"] == 20 and Gx[i]["turns"] == 20]
        if ids:
            d = [Gx[i]["prompt_tokens"] - Lx[i]["prompt_tokens"] for i in ids]
            pos = sum(1 for x in d if x > 0)
            A(f"\n### Payload asymmetry (matched-turn pairs, both at 20 turns)\n")
            A(f"- n = {len(ids)} pairs")
            A(f"- linear mean prompt tokens: "
              f"{st.mean(Lx[i]['prompt_tokens'] for i in ids):.0f}")
            A(f"- graph  mean prompt tokens: "
              f"{st.mean(Gx[i]['prompt_tokens'] for i in ids):.0f}")
            A(f"- median delta (graph - linear): **{st.median(d):+.0f}** tokens")
            A(f"- graph > linear in {pos}/{len(d)}; sign test p = "
              f"**{binom_two_sided(pos, len(d)-pos):.4f}**")

    # thinking-token separation
    A("\n### Reasoning effort by outcome (thinking tokens per turn, est.)\n")
    A("| run_id | success | failure | ratio |")
    A("|---|---|---|---|")
    for rid in runs:
        rs = byrun[rid]
        s = [r for r in rs if r[METRIC]]
        f_ = [r for r in rs if not r[METRIC]]
        if s and f_:
            a = st.mean(r["thinking_tokens_est"] / max(r["turns"], 1) for r in s)
            b = st.mean(r["thinking_tokens_est"] / max(r["turns"], 1) for r in f_)
            A(f"| {rid} | {a:.0f} | {b:.0f} | {a/b if b else 0:.2f}x |")

    with open(args.out, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
