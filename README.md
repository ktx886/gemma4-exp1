# The Efficiency Trap: When Cheaper Retrieval Is Not Better Retrieval

Gemma 4 Developer Agent, Paper Track.

Code graphs and embeddings are offered as context-efficient alternatives to shell
exploration for coding agents. The premise is that bounded structural retrieval is
better retrieval: fewer tokens, less noise, more room to reason.

That premise implies something testable. If cheaper retrieval is better retrieval,
improving efficiency should improve fault localization.

Across 128 SWE-bench-style tasks, with the model, seed, context window, turn budget
and scoring held fixed, it does not. Efficiency has a threshold, not a slope.

| | |
|---|---|
| **Paper and writeup** | https://www.kaggle.com/competitions/gemma-4-developer-agent |
| **Analysis notebook** | https://www.kaggle.com/code/kmd1111/the-efficiency-trap-reproducing-the-analysis |
| **Raw results** | https://www.kaggle.com/datasets/kmd1111/efficiency-trap-agent-traces |

This repository holds the code. The experimental output, roughly 2,000 per-turn
trace files across eight sweeps, lives in the Kaggle dataset above, and the notebook
recomputes every number in the paper from it.

---

## Results

Six conditions, same 128 tasks, same model (Gemma 4 31B at 4-bit), same seed (1337),
same 16,384-token window, same 20-turn budget. Only the retrieval tools and their
payload differ.

| Run | What changes | loc_strict | loc_any | Truncated |
|---|---|---|---|---|
| **S1** | shell only, the baseline | **54.7%** | **76.6%** | 7.8% |
| **S4c** | hybrid, shell withheld turns 0-4 | 49.2% | 72.7% | 8.6% |
| **S4** | hybrid, free choice | 47.7% | 73.4% | 14.1% |
| **S2a** | graph, 5 results x 200 chars | 35.9% | 62.5% | 24.2% |
| **S5** | graph, signatures on navigation | 33.6% | 65.6% | 17.2% |
| **S2b** | graph, 10 results x 1,200 chars | 3.1% | 15.6% | 76.6% |

`loc_strict` means the answer names a file **and** a symbol the maintainer's patch
actually changed. `loc_any` means either.

Three interventions moved efficiency. One moved accuracy.

| Intervention | Efficiency | Accuracy | p |
|---|---|---|---|
| fat to lean payload | truncation 76.6% to 24.2% | **+32.8 points** | < 10⁻⁶ |
| signatures on navigation | tokens −7%, exhaustion −29% | −2.3 points | 0.69 |
| staged tool access | runtime −27%, truncation −5.5pp | +1.6 points | 0.87 |

Comparisons are paired by task and tested with McNemar's exact test.

---

## Contents

```
exp1/
  harness_v2.py            the agent harness: all conditions, tool
                           implementations, per-turn tracing
  run_sweep.sh             sweep orchestrator, with the post-run guard that
                           verifies logged config against requested config
  score_localization.py    deterministic scorer; no LLM judge
  audit_graphs.py          CPU-only audit of the released graphs and embeddings
  analyze.py               aggregation across runs
  verify_resolution.py     pre-flight: do gold symbols resolve in our tools?
  test_orphan_reaping.py   regression test for process-group termination
  smoke.sh                 generic pre-flight for any condition
  smoke_s4c.sh             shell-lock enforcement test
  smoke_s5.sh              signature-payload check
```

---

## Reproducing

Obtain the task set, repository snapshots, code graphs and embedding archives from
the competition page, then:

```bash
# pre-flight, CPU only, no model required
python3 exp1/verify_resolution.py --dataset <path>
python3 exp1/audit_graphs.py      --dataset <path> --out audit/

# the six reported conditions
SNIPPET_LIMIT=200  SEARCH_K=5  ./exp1/run_sweep.sh linear 1337
SNIPPET_LIMIT=200  SEARCH_K=5  ./exp1/run_sweep.sh graph  1337 lean
SNIPPET_LIMIT=1200 SEARCH_K=10 ./exp1/run_sweep.sh graph  1337 fat
SNIPPET_LIMIT=200  SEARCH_K=5  ./exp1/run_sweep.sh hybrid 1337 lean
WITH_SIGNATURES=1 SNIPPET_LIMIT=200 SEARCH_K=5 ./exp1/run_sweep.sh graph 1337 sig2
SHELL_LOCK_LEN=5  SNIPPET_LIMIT=200 SEARCH_K=5 ./exp1/run_sweep.sh hybrid 1337 shelllock

# score
python3 exp1/score_localization.py --dataset <path> --runs results/ \
    --out scored_all.csv
```

Note: The code assumes Gemma4 model served at localhost:11434 and the dataset is downloaded and available locally.

Runs are resumable: the harness skips any (instance, condition) pair already present
in `results.jsonl`.

`run_sweep.sh` compares the configuration logged on its result rows against the
configuration requested on the command line and warns on mismatch. That guard exists
because a computed-but-unpassed flag once produced a silent duplicate of another
condition, which cost a full sweep to discover.

Hardware used: one Radeon Pro W7900 (48 GB), Ryzen 9 7950X3D, 125 GB RAM, Ollama
0.31.1. Any machine that can serve a 31B model at 4-bit will do.

---

## Known issue

`audit_graphs.py` populates `gold_in_snapshot` only when run without
`--skip-snapshots`. If that flag is used, the field stays empty and section A6 of the
generated summary reports `0/129`, which is an artifact rather than a measurement.
The audit shipped in the Kaggle dataset was produced with the flag, so A6 should be
disregarded there; the notebook omits it. No claim in the paper depends on it.

---

## Competition data

No competition data is redistributed in this repository or in the linked dataset. The
task set, repository snapshots, code graphs and embedding archives are obtained
directly from Kaggle.

## License

Apache 2.0. See `LICENSE`.

