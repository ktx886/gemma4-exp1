#!/usr/bin/env python3
"""Score agent answers by fault localization against the gold patch.

This measures FAULT LOCALIZATION, not patch correctness. An answer scores a
hit when it names a file or symbol that the gold patch actually modifies.
Deterministic, no LLM judge, reproducible from the public dataset.

    loc_file   : answer names a gold-patch file (full path or basename)
    loc_symbol : answer names a gold-patch function/class
    loc_strict : both

Usage:
    python3 score_localization.py --dataset DS --results results/r1 [results/r2 ...] \
        --out scored.csv
"""

import argparse
import collections
import csv
import json
import os
import re


# Path prefixes that are test/doc scaffolding rather than library source. A gold
# patch often touches both; symbols defined only in the scaffolding (test
# fixtures, tutorial snippets) are not what the agent is asked to localize.
NON_SOURCE_PREFIXES = ("tests/", "test/", "testing/", "docs/", "docs_src/",
                       "doc/", "scripts/", "benchmarks/", "benchmark/",
                       "examples/", "example/", "tools/")


def _is_source_path(path):
    p = path.lstrip("./")
    return not any(p.startswith(x) or ("/" + x) in p for x in NON_SOURCE_PREFIXES)


def _patch_sections(patch):
    """Split a unified diff into (filepath, body) sections."""
    out, cur, buf = [], None, []
    for line in patch.splitlines():
        m = re.match(r"^\+\+\+ b/(.+?)\s*$", line)
        if m:
            if cur is not None:
                out.append((cur, "\n".join(buf)))
            cur, buf = m.group(1), []
        elif cur is not None:
            buf.append(line)
    if cur is not None:
        out.append((cur, "\n".join(buf)))
    return out


def gold_files(patch, source_only=True):
    fs = set(re.findall(r"^\+\+\+ b/(.+?)\s*$", patch, re.M))
    fs |= set(re.findall(r"^--- a/(.+?)\s*$", patch, re.M))
    fs = {f for f in fs if f != "/dev/null"}
    if source_only:
        src = {f for f in fs if _is_source_path(f)}
        if src:
            return src
    return fs


def gold_symbols(patch, source_only=True):
    """def/class names introduced or modified by the patch.

    Symbols are collected per diff section so that test fixtures and tutorial
    snippets can be excluded: a patch touching both `fastapi/routing.py` and
    `docs_src/.../tutorial.py` would otherwise contribute `read_users` (a test
    endpoint) as a gold symbol, and an answer correctly naming the library
    function would be scored a miss.
    """
    def _syms(body):
        s = set(re.findall(r"^@@.*@@\s*(?:async\s+)?(?:def|class)\s+(\w+)",
                           body, re.M))
        s |= set(re.findall(r"^[+-]\s*(?:async\s+)?(?:def|class)\s+(\w+)",
                            body, re.M))
        return {x for x in s if len(x) > 2 and not x.startswith("__")}

    sections = _patch_sections(patch)
    all_syms, src_syms = set(), set()
    for path, body in sections:
        s = _syms(body)
        all_syms |= s
        if _is_source_path(path):
            src_syms |= s
    if source_only and src_syms:
        return src_syms
    return all_syms


def score_answer(text, files, syms):
    """Return (file_hit, symbol_hit, matched_file, matched_symbol).

    A file counts as named only if the answer gives a path-qualified reference:
    the full path, or at least `parent/name.py`. A bare basename is not
    sufficient -- `utils.py` occurs in most Python repositories and would score
    a hit without the agent having localized anything.
    """
    if not text.strip():
        return 0, 0, "", ""
    low = text.lower()
    mf = ""
    for f in sorted(files, key=len, reverse=True):
        fl = f.lower()
        if fl in low:
            mf = f
            break
        # accept the last two path components, e.g. 'requests/utils.py'
        parts = f.split("/")
        if len(parts) >= 2:
            tail = "/".join(parts[-2:]).lower()
            if tail in low:
                mf = tail
                break
    ms = ""
    for s in sorted(syms, key=len, reverse=True):
        if re.search(rf"\b{re.escape(s)}\b", text):
            ms = s
            break
    return int(bool(mf)), int(bool(ms)), mf, ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--results", nargs="+", required=True,
                    help="one or more run directories (containing results.jsonl)")
    ap.add_argument("--out", default="scored.csv")
    args = ap.parse_args()

    with open(os.path.join(args.dataset, "tasks.jsonl"), encoding="utf8") as f:
        tasks = {t["instance_id"]: t for t in (json.loads(l) for l in f if l.strip())}
    gold = {k: (gold_files(v["patch"]), gold_symbols(v["patch"]))
            for k, v in tasks.items()}

    rows = []
    for rd in args.results:
        rp = os.path.join(rd, "results.jsonl")
        if not os.path.exists(rp):
            print(f"!! no results.jsonl in {rd}")
            continue
        with open(rp) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                inst, cond = r["instance_id"], r["condition"]
                ap_ = os.path.join(rd, "answers", f"{inst}_{cond}.md")
                text = ""
                if os.path.exists(ap_):
                    with open(ap_, errors="replace") as af:
                        text = af.read()
                files, syms = gold.get(inst, (set(), set()))
                fh, sh, mf, ms = score_answer(text, files, syms)
                r.update({
                    "answer_present": int(os.path.exists(ap_)),
                    "answer_len": len(text),
                    "n_gold_files": len(files), "n_gold_symbols": len(syms),
                    "loc_file": fh, "loc_symbol": sh,
                    "loc_strict": int(fh and sh), "loc_any": int(fh or sh),
                    "matched_file": mf, "matched_symbol": ms,
                })
                rows.append(r)

    if not rows:
        print("no rows scored")
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    print(f"scored {len(rows)} runs -> {args.out}\n")
    g = collections.defaultdict(list)
    for r in rows:
        g[(r.get("run_id", "?"), r["condition"])].append(r)
    print(f"{'run / condition':42}{'n':>5}{'loc_file':>10}{'loc_sym':>9}"
          f"{'strict':>8}{'answered':>10}")
    for k in sorted(g):
        rs = g[k]
        n = len(rs)
        ans = sum(1 for r in rs if r["stop_reason"] == "final_answer")
        print(f"{(k[0]+' / '+k[1])[:40]:42}{n:5d}"
              f"{100*sum(r['loc_file'] for r in rs)/n:9.1f}%"
              f"{100*sum(r['loc_symbol'] for r in rs)/n:8.1f}%"
              f"{100*sum(r['loc_strict'] for r in rs)/n:7.1f}%"
              f"{100*ans/n:9.1f}%")


if __name__ == "__main__":
    main()
