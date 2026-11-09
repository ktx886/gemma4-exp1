#!/usr/bin/env python3
"""Evaluation harness v2 for the Gemma 4 Developer Agent experiment.

Three tool conditions against one Ollama-served model:

  linear : run_bash_command                                  (filesystem retrieval)
  graph  : search_similar_code + get_code_neighbors          (structural retrieval)
  hybrid : all three                                         (can the model self-route?)

search_similar_code mirrors the official swegemma semantics: the query is
resolved to a graph node, then that node's nearest neighbours by cosine
similarity over the provided 256-d embeddings are returned. It is NOT a
natural-language keyword search.

Changes vs v1 (all logged, all reproducible):
  F1  answers always written, even when empty
  F2  per-task exception guard: one bad task cannot kill the sweep
  F3  max_neighbors coerced once (v1 compared a possibly-str to an int)
  F4  explicit stop_reason; no magic answer-length sentinel
  F5  explicit options block (num_ctx/temperature/top_p/top_k/seed) + logged
  F6  per-turn JSONL trace
  F7  done_reason captured per turn
  F8  /proc/meminfo AnonPages vs Cached + harness RSS, sampled per turn
  F9  run_id on every row
  F10 setup_s separated from agent_s
  F11 real embedding search (was a BM25 keyword proxy)
  F12 neighbors_returned / results_returned logged per tool call

Usage:
  python3 harness_v2.py --dataset DS --condition linear --out results/r1 \
      --model gemma4:31b --num-ctx 16384 --seed 1337
"""

import argparse
import collections
import json
import os
import re
import signal
try:
    import resource                                   # POSIX only
except ImportError:                                   # pragma: no cover
    resource = None
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback

import numpy as np
import requests

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

BASH_OUTPUT_LIMIT = 5000      # matches the official harness max_stdout_chars
GRAPH_OUTPUT_LIMIT = 5000     # same cap for graph tools
BASH_TIMEOUT_S = 30

# Per-result source budget for search_similar_code. The official harness divides
# its whole 5,000-char allowance across k results, which at k=10 makes every
# call cost the full budget. That is viable at the official 32,768-token total
# budget but saturates a 16,384-token input window in ~12 turns. We therefore
# expose the per-result cap as the experiment's independent variable:
#   400  -> "lean", ~1.5k chars/call  (the regime concurrent work reports)
#   1200 -> "fat",  ~5k chars/call    (the official budget read literally)
DEFAULT_SEARCH_SNIPPET_LIMIT = 400
DEFAULT_SEARCH_K = 10         # official default; lowered in the lean condition

# The official graph API splits navigation from content: get_code_neighbors
# returns names, and seeing any source needs a second call to
# search_similar_code. Every hop therefore costs two calls, and the content call
# dominates the context budget. The graph-signature condition tests the obvious
# remedy -- attach each neighbour's def/class line so one call suffices.
SIGNATURE_LIMIT = 120         # chars per neighbour signature

# Only tasks unsolvable in EVERY condition are excluded. audit_graphs.py (A6)
# shows fastapi_15661's gold file is absent from its own snapshot, so neither
# bash nor graph navigation can reach it. Tasks that are merely graph-
# unreachable are KEPT: dropping them would bias the comparison against the
# linear condition. They are flagged during analysis instead.
DEFAULT_EXCLUDE = {"fastapi_15661"}

NON_SOURCE_NS = {"tests", "test", "testing", "docs", "docs_src", "doc",
                 "scripts", "benchmarks", "benchmark", "examples", "example", "tools"}


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

def ollama_chat(model, messages, tools, options, host, timeout=900):
    payload = {"model": model, "messages": messages, "stream": False,
               "options": options}
    if tools:
        payload["tools"] = tools
    r = requests.post(f"{host}/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "run_bash_command",
        "description": (
            "Run a shell command (grep, cat, ls, find, ...) inside the "
            "repository checkout. Returns combined stdout/stderr, truncated "
            "to 5000 characters."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "cmd": {"type": "string", "description": "The shell command to execute."}
            },
            "required": ["cmd"],
        },
    },
}

NEIGHBORS_TOOL = {
    "type": "function",
    "function": {
        "name": "get_code_neighbors",
        "description": (
            "Find callers and callees of a symbol node (e.g. "
            "'pkg.module.Class.method' or 'Class.method'). Returns the names of "
            "connected nodes. Use search_similar_code to see their source."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "node": {"type": "string", "description": "Symbol to expand."},
                "edge_type": {"type": "string",
                              "description": "Optional filter: 'calls' or 'called_by'."},
                "max_neighbors": {"type": "integer",
                                  "description": "Maximum neighbors (default 50)."},
            },
            "required": ["node"],
        },
    },
}

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_similar_code",
        "description": (
            "Find functions/classes with similar precomputed embeddings given a "
            "symbol name (e.g. 'pkg.module.Class.method', 'Class.method', or "
            "'func_name'). Returns node names with their source code."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Symbol name to search from."},
                "k": {"type": "integer", "description": "Number of results (default 10)."},
            },
            "required": ["query"],
        },
    },
}

SUBGRAPH_TOOL = {
    "type": "function",
    "function": {
        "name": "get_code_subgraph",
        "description": (
            "Get the induced call subgraph connecting a list of symbol nodes. "
            "Use this to see how several symbols you have found relate to each "
            "other."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "nodes": {"type": "array", "items": {"type": "string"},
                          "description": "Symbol names to connect."},
            },
            "required": ["nodes"],
        },
    },
}

SUBMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_answer",
        "description": (
            "Submit your final answer and end the session. Call this as soon as "
            "you can name the file(s) and function(s) that must change. Do not "
            "keep investigating once you know the answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": (
                        "The final answer: the file path(s) and symbol name(s) "
                        "that must change, the root cause, and the exact code "
                        "changes needed."
                    ),
                }
            },
            "required": ["answer"],
        },
    },
}

TOOLSETS = {
    "linear": [BASH_TOOL, SUBMIT_TOOL],
    "graph": [SEARCH_TOOL, NEIGHBORS_TOOL, SUBGRAPH_TOOL, SUBMIT_TOOL],
    "hybrid": [BASH_TOOL, SEARCH_TOOL, NEIGHBORS_TOOL, SUBGRAPH_TOOL, SUBMIT_TOOL],
}

_SIG_DESC = (
    "Find callers and callees of a symbol node (e.g. 'pkg.module.Class.method' "
    "or 'Class.method'). Returns each connected node with its def/class "
    "signature, so you can choose where to go next without a separate lookup."
)


def toolset(condition, with_signatures=False):
    """Tool list for a condition. With signatures enabled, get_code_neighbors
    advertises that it returns signatures, not bare names -- the model must know
    the tool's contract changed or it will keep reaching for the content tool."""
    tools = TOOLSETS[condition]
    if not with_signatures:
        return tools
    out = []
    for t in tools:
        if t["function"]["name"] == "get_code_neighbors":
            t = json.loads(json.dumps(t))
            t["function"]["description"] = _SIG_DESC
        out.append(t)
    return out

# Turns remaining at which the model starts being warned to finalize.
BUDGET_WARN_AT = 5


def shell_locked(turn, lock_len, cycle_len):
    """True when run_bash_command is withheld on this turn (condition S4c).

    lock_len=0 disables the mechanism. With lock_len=5 and cycle_len=10 the
    shell is unavailable on turns 0-4, 10-14, 20-24 ... so the agent is forced
    back to structural navigation periodically rather than once at the start,
    which it could otherwise simply wait out.
    """
    if lock_len <= 0:
        return False
    if cycle_len <= 0:
        return turn < lock_len
    return (turn % cycle_len) < lock_len


def truncate(text, limit):
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[... truncated, {len(text) - limit} chars omitted]"


def truncate_plain(text, limit):
    """Hard cut with no annotation -- for short fields such as signatures."""
    return text if len(text) <= limit else text[:limit - 3] + "..."


# ---------------------------------------------------------------------------
# linear condition
# ---------------------------------------------------------------------------

class Workspace:
    def __init__(self, snapshot_path):
        self._tmp = tempfile.TemporaryDirectory(prefix="harness_ws_")
        self._groups = []          # pgids spawned by this workspace
        with tarfile.open(snapshot_path) as tf:
            tf.extractall(self._tmp.name, filter="data")

    def run_bash_command(self, cmd):
        """Run a shell command, reaping the whole process group on timeout.

        subprocess.run(shell=True, timeout=...) kills only the shell. Any
        process the shell started survives, reparents to init, and keeps
        running after the harness exits. In an earlier version of this harness
        that produced three orphaned `python3 reproduce_issue.py` processes
        that ran at 99.9% CPU for up to five days across four separate sweeps,
        and one that allocated to 121 GB (section 7). start_new_session=True
        puts the shell in its own process group so os.killpg can take the
        children with it.
        """
        proc = subprocess.Popen(
            cmd, shell=True, cwd=self._tmp.name,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True,
        )
        try:
            self._groups.append(os.getpgid(proc.pid))
        except (ProcessLookupError, OSError):
            pass
        try:
            stdout, stderr = proc.communicate(timeout=BASH_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            self._kill_group(proc)
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                stdout, stderr = "", ""
            partial = (stdout or "")[:BASH_OUTPUT_LIMIT]
            return (f"[error] command timed out after {BASH_TIMEOUT_S}s "
                    f"and its process group was terminated"
                    + (f"\n[partial output]\n{partial}" if partial.strip() else "")), 0

        out = stdout or ""
        if stderr:
            out += ("\n" if out else "") + "[stderr]\n" + stderr
        if not out.strip():
            out = f"[no output, exit code {proc.returncode}]"
        return truncate(out, BASH_OUTPUT_LIMIT), out.count("\n") + 1

    @staticmethod
    def _kill_group(proc):
        """SIGTERM the process group, then SIGKILL anything still alive."""
        try:
            pgid = os.getpgid(proc.pid)
        except (ProcessLookupError, OSError):
            return
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pgid, sig)
            except (ProcessLookupError, OSError):
                return
            try:
                proc.wait(timeout=2)
                return
            except subprocess.TimeoutExpired:
                continue

    def close(self):
        # Backstop: nothing this workspace started may outlive it.
        for pgid in self._groups:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
        self._groups.clear()
        self._tmp.cleanup()


# ---------------------------------------------------------------------------
# graph condition  (F11: real embeddings, official semantics)
# ---------------------------------------------------------------------------

class GraphTools:
    def __init__(self, graph_path, emb_path,
                 snippet_limit=DEFAULT_SEARCH_SNIPPET_LIMIT,
                 default_k=DEFAULT_SEARCH_K,
                 with_signatures=False):
        self.snippet_limit = int(snippet_limit)
        self.default_k = int(default_k)
        self.with_signatures = bool(with_signatures)
        with open(graph_path) as f:
            g = json.load(f)
        self.nodes = {n["id"]: n for n in g["nodes"]}
        self.ids = list(self.nodes)
        self._lower = {i: i.lower() for i in self.ids}
        self.adj = collections.defaultdict(list)
        for e in g["edges"]:
            if e.get("type") != "calls":
                continue
            self.adj[e["source"]].append(("calls", e["target"]))
            self.adj[e["target"]].append(("called_by", e["source"]))

        self.emb_keys, self.emb_mat, self.emb_index = [], None, {}
        if emb_path and os.path.exists(emb_path):
            z = np.load(emb_path, allow_pickle=True)
            self.emb_keys = list(z.keys())
            if self.emb_keys:
                m = np.stack([z[k] for k in self.emb_keys]).astype(np.float32)
                n = np.linalg.norm(m, axis=1, keepdims=True)
                n[n == 0] = 1.0
                self.emb_mat = m / n
                self.emb_index = {k: i for i, k in enumerate(self.emb_keys)}

    def _resolve(self, name, pool, lower_map=None):
        """Official 4-tier resolution: exact -> suffix -> case-insens -> substring."""
        if name in pool:
            return name
        for k in pool:
            if k.endswith("." + name):
                return k
        low = name.lower()
        for k in pool:
            if (lower_map[k] if lower_map else k.lower()) == low:
                return k
        for k in pool:
            if low in (lower_map[k] if lower_map else k.lower()):
                return k
        return None

    def _resolve_emb(self, name):
        """Resolve a symbol to an embedding key.

        The graph keyspace and the embedding keyspace are not identical: some
        symbols present as graph nodes are absent from the .npz. Resolve against
        the embeddings first, then fall back to resolving in the graph and
        re-resolving that node id (and its trailing components) in the
        embeddings. Without this, bare function names the model legitimately
        asks for silently return zero hits.
        """
        key = self._resolve(name, self.emb_keys)
        if key is not None:
            return key
        node = self._resolve(name, self.ids, self._lower)
        if node is None:
            return None
        key = self._resolve(node, self.emb_keys)
        if key is not None:
            return key
        parts = node.split(".")
        for i in range(1, len(parts)):
            key = self._resolve(".".join(parts[i:]), self.emb_keys)
            if key is not None:
                return key
        return None

    def _signature(self, node_id):
        """First meaningful line of a node's source: its def/class signature.

        Collapses a multi-line signature onto one line and caps the result, so
        the cost is bounded regardless of how long the parameter list is.
        """
        txt = self.nodes.get(node_id, {}).get("text", "")
        if not txt:
            return ""
        lines = [l for l in txt.splitlines() if l.strip()
                 and not l.lstrip().startswith(("#", "@"))]
        if not lines:
            return ""
        sig = lines[0].strip()
        # A signature split over several lines has an unbalanced '('.
        i = 1
        while sig.count("(") > sig.count(")") and i < len(lines) and i < 8:
            sig += " " + lines[i].strip()
            i += 1
        return truncate_plain(sig, SIGNATURE_LIMIT)

    def get_code_neighbors(self, node, edge_type=None, max_neighbors=50):
        """Callers and callees of a symbol.

        Two serialisations, selected by self.with_signatures:

        names-only (official swegemma semantics)
            {"neighbors": ["calls -> pkg.mod.fn", ...]}
            Navigation only. The agent must call search_similar_code to see any
            source, so every hop through the graph costs two tool calls.

        with signatures (the one-call variant this paper proposes)
            {"neighbors": [{"edge": "calls", "node": "pkg.mod.fn",
                            "signature": "def fn(a, b=None):"}, ...]}
            Adds the def/class line for each neighbour, so the agent can choose
            where to go next without a second call. Costs a few hundred
            characters more per call, and is the intervention tested in the
            graph-signature condition.
        """
        try:
            mx = int(max_neighbors)
        except (TypeError, ValueError):
            mx = 50
        mx = max(1, min(mx, 200))

        resolved = self._resolve(str(node), self.ids, self._lower)
        if resolved is None:
            hint = ""
            if self.emb_mat is not None:
                near = self.search_similar_code(str(node), k=5, _raw=True)
                if near:
                    hint = "\nDid you mean:\n" + "\n".join(f"  - {n}" for n, _ in near)
            return (json.dumps({"status": "error", "error_type": "NodeNotFound",
                                "error_message": f"symbol '{node}' not found"})
                    + hint), 0

        edges = self.adj.get(resolved, [])
        if edge_type:
            want = str(edge_type).lower()
            edges = [(d, n) for d, n in edges if want in ("calls", "called_by", d)]
        if not edges:
            return json.dumps({"status": "ok", "node": resolved, "neighbors": [],
                               "count": 0,
                               "note": "node exists but has no 'calls' edges"}), 0

        shown = edges[:mx]
        if self.with_signatures:
            items = [{"edge": d, "node": n, "signature": self._signature(n)}
                     for d, n in shown]
        else:
            items = [f"{d} -> {n}" for d, n in shown]

        def _pack(lst, trunc=False):
            p = {"status": "ok", "node": resolved, "neighbors": lst,
                 "count": len(edges)}
            if trunc:
                p["is_truncated"] = True
            return json.dumps(p)

        out = _pack(items)
        while len(out) > GRAPH_OUTPUT_LIMIT and items:
            items.pop()
            out = _pack(items, trunc=True)
        return out, len(items)

    def search_similar_code(self, query, k=None, _raw=False):
        """Official swegemma semantics: top-k by cosine similarity, each result
        carrying a source snippet, with the 5,000-char budget divided across the
        k results (unused budget from short snippets rolling over to longer
        ones)."""
        try:
            kk = self.default_k if k is None else int(k)
        except (TypeError, ValueError):
            kk = self.default_k
        kk = max(1, min(kk, 50))

        if self.emb_mat is None:
            return json.dumps({"status": "error", "error_type": "NoEmbeddings",
                               "error_message": "no embeddings for this repo"}), 0
        key = self._resolve_emb(str(query))
        if key is None:
            if _raw:
                return []
            return json.dumps({
                "status": "error", "error_type": "QueryUnresolved",
                "error_message": (f"'{query}' could not be resolved to a node; "
                                  f"pass a symbol such as 'Class.method' or "
                                  f"'function_name'")}), 0

        i = self.emb_index[key]
        sims = self.emb_mat @ self.emb_mat[i]
        sims[i] = -2.0
        kk = min(kk, len(sims) - 1)
        top = np.argpartition(-sims, kk)[:kk]
        top = top[np.argsort(-sims[top])]
        hits = [(self.emb_keys[j], float(sims[j])) for j in top]
        if _raw:
            return hits

        # Per-result source budget -- the experiment's independent variable.
        # A hard per-result cap, NOT the official rollover: rollover lets a few
        # long functions absorb the whole 5,000-char allowance, which makes the
        # cap inoperative (every call costs the full budget regardless of the
        # setting). Total is additionally capped at GRAPH_OUTPUT_LIMIT.
        per = self.snippet_limit
        results = []
        for nid, s in hits:
            txt = self.nodes.get(nid, {}).get("text", "")
            results.append({"node_name": nid, "code": txt[:per],
                            "similarity": round(s, 4)})
        out = json.dumps({"status": "ok", "query": key, "results": results,
                          "count": len(results)})
        return truncate(out, GRAPH_OUTPUT_LIMIT), len(hits)

    def get_code_subgraph(self, nodes):
        """Official swegemma tool 9: induced call subgraph over a node list."""
        if isinstance(nodes, str):
            try:
                nodes = json.loads(nodes)
            except json.JSONDecodeError:
                nodes = [x.strip() for x in nodes.split(",") if x.strip()]
        if not isinstance(nodes, list):
            nodes = [str(nodes)]

        resolved, missing = [], []
        for n in nodes[:50]:
            r = self._resolve(str(n), self.ids, self._lower)
            (resolved if r else missing).append(r or str(n))
        rset = set(resolved)
        edges = []
        for src in rset:
            for d, tgt in self.adj.get(src, []):
                if d == "calls" and tgt in rset:
                    edges.append({"from": src, "to": tgt, "type": "calls"})
        payload = {"status": "ok", "nodes": resolved, "edges": edges,
                   "node_count": len(resolved), "edge_count": len(edges)}
        if missing:
            payload["unresolved"] = missing
        return truncate(json.dumps(payload), GRAPH_OUTPUT_LIMIT), len(edges)


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------

# Mirrors the official harness "Execution Environment Rules": the sandbox is
# air-gapped and fully provisioned. Without this, agents burn several turns on
# pip install / venv creation that can never succeed.
ENV_RULES = (
    "\n\nEnvironment rules:\n"
    "- The environment is OFFLINE. There is no network or PyPI access. All "
    "repository and test dependencies are ALREADY installed. Do NOT run "
    "pip install, create virtualenvs, or download anything -- those commands "
    "will fail and waste your budget.\n"
    "- Command output is truncated to 5000 characters.\n"
    "- You have a limited number of turns. Localizing the fault is the goal; "
    "you do not need to execute or verify a fix."
)

COMMON_TAIL = (
    " Investigate the problem, then call submit_answer with a final answer that "
    "names the file(s) and function(s) that must change, explains the root "
    "cause, and gives the exact code changes needed. State file paths and "
    "symbol names explicitly. Call submit_answer as soon as you can identify "
    "the faulty code -- do not keep exploring once you know the answer."
)

SYSTEM_PROMPTS = {
    "linear": (
        "You are a software engineering agent. You are given a bug report or "
        "feature request for a Python repository checked out in your working "
        "directory. Use the run_bash_command tool (ls, find, grep, cat, ...) to "
        "explore the code." + COMMON_TAIL + ENV_RULES
    ),
    "graph": (
        "You are a software engineering agent. You are given a bug report or "
        "feature request for a Python repository. You cannot run commands or "
        "read files. This repository has pre-built code graph and embedding "
        "data indexed by dotted symbol path (e.g. pkg.module.Class.method). "
        "Use these tools for fast, targeted navigation:\n"
        "- search_similar_code(query): find functions/classes with similar "
        "embeddings given a symbol name; returns their source code.\n"
        "- get_code_neighbors(node): find callers and callees of a symbol; "
        "returns node names.\n"
        "- get_code_subgraph(nodes): get the induced call subgraph connecting "
        "a list of symbols." + COMMON_TAIL + ENV_RULES
    ),
    "hybrid": (
        "You are a software engineering agent. You are given a bug report or "
        "feature request for a Python repository checked out in your working "
        "directory. You have two kinds of tools: run_bash_command for shell "
        "exploration (ls, find, grep, cat), and code-graph tools for "
        "structural navigation by symbol (search_similar_code, "
        "get_code_neighbors, get_code_subgraph). The repository has pre-built "
        "graph and embedding data indexed by dotted symbol path. Choose "
        "whichever is most effective at each step." + COMMON_TAIL + ENV_RULES
    ),
}

# Alternative hybrid framing (condition S4b). S4's prompt lists the shell first
# and is neutral between the toolsets; the agent used the shell for 99.3% of
# calls. This variant inverts the order and actively recommends the graph tools,
# so that "the naive composition degenerates to shell-only" can be separated
# from "we framed the choice badly". Everything except this text is identical.
SYSTEM_PROMPTS_GRAPHFIRST = dict(SYSTEM_PROMPTS)
SYSTEM_PROMPTS_GRAPHFIRST["hybrid"] = (
    "You are a software engineering agent. You are given a bug report or "
    "feature request for a Python repository. This repository has pre-built "
    "code-graph and embedding data indexed by dotted symbol path, and these "
    "tools are the fastest way to locate a symbol and understand how it is "
    "used: search_similar_code finds functions resembling a symbol, "
    "get_code_neighbors finds what calls it and what it calls, and "
    "get_code_subgraph shows how several symbols connect. Prefer these tools "
    "whenever you are looking for a specific function or tracing a "
    "dependency. A shell is also available through run_bash_command (ls, "
    "find, grep, cat) for reading a file directly or searching for text that "
    "is not a symbol name." + COMMON_TAIL + ENV_RULES
)

PROMPT_VARIANTS = {
    "default": SYSTEM_PROMPTS,
    "graphfirst": SYSTEM_PROMPTS_GRAPHFIRST,
}


def task_prompt(task):
    parts = [f"Repository: {task['repo']}", "", task["problem_statement"].strip()]
    hints = (task.get("hints_text") or "").strip()
    if hints:
        parts += ["", "Hints:", hints]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# F8: memory sampling
# ---------------------------------------------------------------------------

def mem_sample():
    out = {"anon_gb": "", "cached_gb": "", "memfree_gb": "", "rss_gb": ""}
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, _, v = line.partition(":")
                info[k] = float(v.strip().split()[0]) / (1024 * 1024)
        out["anon_gb"] = round(info.get("AnonPages", 0.0), 3)
        out["cached_gb"] = round(info.get("Cached", 0.0), 3)
        out["memfree_gb"] = round(info.get("MemFree", 0.0), 3)
    except Exception:                                        # noqa: BLE001
        pass
    if resource is not None:
        try:
            out["rss_gb"] = round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024), 3)
        except Exception:                                    # noqa: BLE001
            pass
    return out


# ---------------------------------------------------------------------------
# agent loop
# ---------------------------------------------------------------------------

def run_agent(task, condition, ds, model, max_turns, host, options,
              run_id, answers_dir, traces_dir,
              snippet_limit=DEFAULT_SEARCH_SNIPPET_LIMIT,
              search_k=DEFAULT_SEARCH_K, with_signatures=False,
              prompt_variant="default", shell_lock_len=0, shell_lock_cycle=0):
    inst = task["instance_id"]
    t0 = time.time()
    ws = graph = None
    setup_s = 0.0

    pre = inst.rsplit("_", 1)[0]
    if condition in ("linear", "hybrid"):
        ws = Workspace(os.path.join(ds, "snapshots", f"{inst}.tgz"))
    if condition in ("graph", "hybrid"):
        gp = os.path.join(ds, "graphs", f"{pre}_{task['base_commit']}.json")
        ep = os.path.join(ds, "embeddings", f"{pre}_{task['base_commit']}.npz")
        if not os.path.exists(gp):
            raise FileNotFoundError(f"no graph for {inst} ({task['base_commit']})")
        graph = GraphTools(gp, ep, snippet_limit=snippet_limit,
                           default_k=search_k,
                           with_signatures=with_signatures)
    setup_s = time.time() - t0                               # F10

    def dispatch(name, a, locked=False):
        if name == "run_bash_command" and locked:
            # S4c: the tool is withdrawn from the advertised list on locked
            # turns, but the model still calls it -- the system prompt
            # describes a shell and the history is full of prior shell calls.
            # Withdrawing it from the menu is not enough; it must also be
            # refused at dispatch, or the lock measures nothing.
            return ("[error] run_bash_command is unavailable on this turn. "
                    "Use search_similar_code, get_code_neighbors or "
                    "get_code_subgraph."), 0
        if name == "run_bash_command" and ws is not None:
            return ws.run_bash_command(a.get("cmd", ""))
        if name == "get_code_neighbors" and graph is not None:
            return graph.get_code_neighbors(a.get("node") or a.get("node_id", ""),
                                            a.get("edge_type"),
                                            a.get("max_neighbors", 50))
        if name == "search_similar_code" and graph is not None:
            return graph.search_similar_code(a.get("query", ""), a.get("k", 10))
        if name == "get_code_subgraph" and graph is not None:
            return graph.get_code_subgraph(a.get("nodes", []))
        return f"[error] unknown or unavailable tool '{name}'", 0

    tools = toolset(condition, with_signatures)
    prompts = PROMPT_VARIANTS.get(prompt_variant, SYSTEM_PROMPTS)
    messages = [{"role": "system", "content": prompts[condition]},
                {"role": "user", "content": task_prompt(task)}]

    agent_t0 = time.time()
    n_locked_turns = 0
    n_calls = n_turns = 0
    p_tok = c_tok = think_tok = 0
    tool_hist = collections.Counter()
    zero_result_calls = 0
    answer, stop_reason = "", "unknown"
    trace_path = os.path.join(traces_dir, f"{inst}_{condition}.jsonl")

    with open(trace_path, "w") as tf:
        for turn in range(max_turns):
            # S4c: withhold run_bash_command on locked turns so the agent must
            # navigate structurally. No-op when shell_lock_len is 0.
            locked = shell_locked(turn, shell_lock_len, shell_lock_cycle)
            turn_tools = ([t for t in tools
                           if t["function"]["name"] != "run_bash_command"]
                          if locked else tools)
            r = ollama_chat(model, messages, turn_tools, options, host)
            n_turns += 1
            if locked:
                n_locked_turns += 1
            msg = r.get("message", {}) or {}
            pe = int(r.get("prompt_eval_count") or 0)
            ev = int(r.get("eval_count") or 0)
            p_tok += pe
            c_tok += ev
            thinking = msg.get("thinking") or ""
            think_tok += len(thinking) // 4           # rough; exact split unavailable
            calls = msg.get("tool_calls") or []

            rec = {"turn": turn, "prompt_eval_count": pe, "eval_count": ev,
                   "done_reason": r.get("done_reason"),
                   "thinking_chars": len(thinking),
                   "content_chars": len(msg.get("content") or ""),
                   "n_tool_calls": len(calls), "shell_locked": int(locked),
                   "calls": [],
                   **mem_sample()}                           # F8

            if not calls:
                answer = (msg.get("content") or "").strip()
                stop_reason = "final_answer" if answer else "empty_response"
                rec["stop_reason"] = stop_reason
                tf.write(json.dumps(rec) + "\n")
                break

            messages.append({"role": "assistant",
                             "content": msg.get("content", ""),
                             "tool_calls": calls})
            submitted = False
            for call in calls:
                fn = call.get("function", {}) or {}
                a = fn.get("arguments") or {}
                if isinstance(a, str):
                    try:
                        a = json.loads(a)
                    except json.JSONDecodeError:
                        a = {}
                name = fn.get("name", "")

                if name == "submit_answer":
                    answer = str(a.get("answer", "")).strip()
                    submitted = True
                    tool_hist[name] += 1
                    rec["calls"].append({"name": name, "args": {},
                                         "result_chars": len(answer),
                                         "n_returned": 1})
                    continue

                refused = (name == "run_bash_command" and locked)
                res, n_ret = dispatch(name, a, locked)
                n_calls += 1
                tool_hist[name] += 1
                if n_ret == 0:
                    zero_result_calls += 1
                rec["calls"].append({"name": name,
                                     "args": {k: str(v)[:200] for k, v in a.items()},
                                     "result_chars": len(res),
                                     "n_returned": n_ret,
                                     "refused": int(refused)})  # F12
                tm = {"role": "tool", "content": res}
                if call.get("id"):
                    tm["tool_call_id"] = call["id"]
                messages.append(tm)

            if submitted:
                stop_reason = "submitted" if answer else "submitted_empty"
                rec["stop_reason"] = stop_reason
                tf.write(json.dumps(rec) + "\n")
                break

            # Budget warning, mirroring the official harness's late-session nudge.
            left = max_turns - turn - 1
            if 0 < left <= BUDGET_WARN_AT:
                messages.append({
                    "role": "user",
                    "content": (f"[{left} turn(s) remaining] Finalize now: call "
                                f"submit_answer with your best identification of "
                                f"the file(s) and function(s) that must change, "
                                f"even if you are not fully certain."),
                })
            tf.write(json.dumps(rec) + "\n")
        else:
            stop_reason = "max_turns"                          # F4

    agent_s = time.time() - agent_t0
    if ws is not None:
        ws.close()

    # F1: always write, even when empty
    with open(os.path.join(answers_dir, f"{inst}_{condition}.md"), "w") as f:
        f.write(answer)

    return answer, {
        "run_id": run_id, "instance_id": inst, "condition": condition,
        "model": model, "repo": task["repo"].split("/")[-1],
        "stop_reason": stop_reason,
        "setup_s": round(setup_s, 2), "agent_s": round(agent_s, 2),
        "elapsed_s": round(time.time() - t0, 2),
        "tool_calls": n_calls, "turns": n_turns,
        "prompt_tokens": p_tok, "completion_tokens": c_tok,
        "thinking_tokens_est": think_tok,
        "zero_result_calls": zero_result_calls,
        "tool_mix": json.dumps(dict(tool_hist)),
        "answer_chars": len(answer),
        "num_ctx": options.get("num_ctx"), "temperature": options.get("temperature"),
        "top_p": options.get("top_p"), "top_k": options.get("top_k"),
        "seed": options.get("seed"),
        "snippet_limit": snippet_limit, "search_k": search_k,
        "with_signatures": int(with_signatures),
        "prompt_variant": prompt_variant,
        "shell_lock_len": shell_lock_len, "shell_lock_cycle": shell_lock_cycle,
        "locked_turns": n_locked_turns,
        "error": "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--condition", required=True, choices=["linear", "graph", "hybrid"])
    ap.add_argument("--out", required=True, help="results dir for this run")
    ap.add_argument("--model", default="gemma4:31b")
    ap.add_argument("--host", default=OLLAMA_HOST)
    ap.add_argument("--run-id", default="")
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--snippet-limit", type=int, default=DEFAULT_SEARCH_SNIPPET_LIMIT,
                    help="per-result source chars in search_similar_code")
    ap.add_argument("--search-k", type=int, default=DEFAULT_SEARCH_K,
                    help="default k for search_similar_code when the model "
                         "does not pass one")
    ap.add_argument("--with-signatures", action="store_true",
                    help="attach each neighbour's def/class line to "
                         "get_code_neighbors (the one-call variant)")
    ap.add_argument("--prompt-variant", default="default",
                    choices=sorted(PROMPT_VARIANTS),
                    help="'graphfirst' lists the graph tools first and "
                         "recommends them (condition S4b)")
    ap.add_argument("--shell-lock-len", type=int, default=0,
                    help="withhold run_bash_command for this many turns of "
                         "each cycle (condition S4c; 0 disables)")
    ap.add_argument("--shell-lock-cycle", type=int, default=0,
                    help="cycle length for --shell-lock-len; 0 means lock only "
                         "the opening turns")
    ap.add_argument("--sample", type=int, default=0,
                    help="randomly sample N tasks, balanced across repos")
    ap.add_argument("--sample-seed", type=int, default=20261027,
                    help="seed for --sample, so the subset is reproducible")
    ap.add_argument("--stratify", action="store_true",
                    help="interleave tasks by repo so a --limit N prefix is "
                         "repo-balanced instead of file-ordered")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--task-id", action="append")
    ap.add_argument("--exclude-file", default="")
    ap.add_argument("--no-default-exclude", action="store_true")
    args = ap.parse_args()

    run_id = args.run_id or f"{args.condition}_ctx{args.num_ctx}_seed{args.seed}"
    out = args.out
    answers = os.path.join(out, "answers")
    traces = os.path.join(out, "traces")
    for d in (out, answers, traces):
        os.makedirs(d, exist_ok=True)
    results_path = os.path.join(out, "results.jsonl")

    with open(os.path.join(args.dataset, "tasks.jsonl"), encoding="utf8") as f:
        tasks = [json.loads(l) for l in f if l.strip()]

    excl = set() if args.no_default_exclude else set(DEFAULT_EXCLUDE)
    if args.exclude_file and os.path.exists(args.exclude_file):
        with open(args.exclude_file) as f:
            excl |= {l.strip() for l in f if l.strip()}
    tasks = [t for t in tasks if t["instance_id"] not in excl]

    if args.task_id:
        want = set(args.task_id)
        tasks = [t for t in tasks if t["instance_id"] in want]
    if args.stratify:
        byrepo = collections.OrderedDict()
        for t in tasks:
            byrepo.setdefault(t["repo"], []).append(t)
        order, pools = [], list(byrepo.values())
        while any(pools):
            for p in pools:
                if p:
                    order.append(p.pop(0))
        tasks = order
    if args.sample:
        import random
        rng = random.Random(args.sample_seed)
        byrepo = collections.OrderedDict()
        for t in sorted(tasks, key=lambda x: x["instance_id"]):
            byrepo.setdefault(t["repo"], []).append(t)
        for v in byrepo.values():
            rng.shuffle(v)
        picked, pools = [], list(byrepo.values())
        while len(picked) < args.sample and any(pools):
            for pool in pools:
                if pool and len(picked) < args.sample:
                    picked.append(pool.pop())
        tasks = sorted(picked, key=lambda x: x["instance_id"])
    if args.limit:
        tasks = tasks[:args.limit]

    # resumable: skip (instance, condition) already recorded
    done = set()
    if os.path.exists(results_path):
        with open(results_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["instance_id"], r["condition"]))
                except Exception:                            # noqa: BLE001
                    pass
    todo = [t for t in tasks if (t["instance_id"], args.condition) not in done]

    options = {"num_ctx": args.num_ctx, "temperature": args.temperature,
               "top_p": args.top_p, "top_k": args.top_k, "seed": args.seed,
               "num_predict": -1}

    print(f"run_id={run_id} condition={args.condition} model={args.model}")
    print(f"options={json.dumps(options)}")
    print(f"excluded={len(excl)} todo={len(todo)} already_done={len(tasks)-len(todo)}")
    sys.stdout.flush()

    ok = fail = 0
    with open(results_path, "a") as rf:
        for i, t in enumerate(todo, 1):
            inst = t["instance_id"]
            print(f"[{i}/{len(todo)}] {args.condition} {inst} ... ", end="", flush=True)
            try:                                              # F2
                ans, tel = run_agent(t, args.condition, args.dataset, args.model,
                                     args.max_turns, args.host, options, run_id,
                                     answers, traces,
                                     snippet_limit=args.snippet_limit,
                                     search_k=args.search_k,
                                     with_signatures=args.with_signatures,
                                     prompt_variant=args.prompt_variant,
                                     shell_lock_len=args.shell_lock_len,
                                     shell_lock_cycle=args.shell_lock_cycle)
                ok += 1
                print(f"{tel['stop_reason']:14} {tel['elapsed_s']:7.1f}s "
                      f"turns={tel['turns']:2d} calls={tel['tool_calls']:2d} "
                      f"zero={tel['zero_result_calls']:2d} chars={tel['answer_chars']}",
                      flush=True)
            except Exception as e:                            # noqa: BLE001
                fail += 1
                tel = {"run_id": run_id, "instance_id": inst,
                       "condition": args.condition, "model": args.model,
                       "repo": t["repo"].split("/")[-1], "stop_reason": "error",
                       "setup_s": 0, "agent_s": 0, "elapsed_s": 0, "tool_calls": 0,
                       "turns": 0, "prompt_tokens": 0, "completion_tokens": 0,
                       "thinking_tokens_est": 0, "zero_result_calls": 0,
                       "tool_mix": "{}", "answer_chars": 0,
                       "num_ctx": args.num_ctx, "temperature": args.temperature,
                       "top_p": args.top_p, "top_k": args.top_k, "seed": args.seed,
                       "snippet_limit": args.snippet_limit,
                       "search_k": args.search_k,
                       "with_signatures": int(args.with_signatures),
                       "prompt_variant": args.prompt_variant,
                       "shell_lock_len": args.shell_lock_len,
                       "shell_lock_cycle": args.shell_lock_cycle,
                       "locked_turns": 0,
                       "error": f"{type(e).__name__}: {e}"}
                print(f"ERROR {type(e).__name__}: {e}", flush=True)
                traceback.print_exc(file=sys.stderr)
            rf.write(json.dumps(tel) + "\n")
            rf.flush()

    print(f"\ndone: {ok} ok, {fail} errors -> {results_path}")


if __name__ == "__main__":
    main()
