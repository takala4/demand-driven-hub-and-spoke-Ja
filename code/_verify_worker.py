#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Worker process for verify_small.py — runs ONE implementation on ONE case.

Isolated in a subprocess so a crash here does not take down the orchestrator.
Self-aborts with exit code 99 if resident memory exceeds the supplied cap,
dumping the best-so-far solution to disk before exiting.

Usage:
  python _verify_worker.py <impl> <N> <out_path> <mem_cap_mb> [options]

  impl     : module name. "v1" → hubspoke_v1 (paper-faithful); "v2" → hubspoke
             (improved); any other string is imported as-is (for future
             variants such as hubspoke_v3, hubspoke_dfs, etc.).
  N        : number of nodes.
  out_path : path to dump JSON result.
  mem_cap_mb: hard cap on RSS; process self-terminates if exceeded.
"""
from __future__ import annotations

import argparse
import importlib
import io
import json
import os
import sys
import time
from contextlib import redirect_stdout

import numpy as np
import psutil


# Convenience aliases — extend as new variants are added.
IMPL_MODULES = {
    "v1": "hubspoke_v1",  # paper-faithful Soland branch-and-bound
    "v2": "hubspoke",     # improved Soland branch-and-bound
    "v3": "hubspoke_v3",  # VNS stochastic heuristic
    "v4": "hubspoke_v4",  # Gurobi MILP + PWL (deterministic epsilon-optimal)
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("impl")
    ap.add_argument("N", type=int)
    ap.add_argument("out_path")
    ap.add_argument("mem_cap_mb", type=float)
    ap.add_argument("--phi",  type=float, default=50.0)
    ap.add_argument("--rho",  type=float, default=20.0)
    ap.add_argument("--t",    type=float, default=1.0)
    ap.add_argument("--nu",   type=float, default=10.0)
    ap.add_argument("--d",    type=float, default=1.0)
    ap.add_argument("--max-leaves", type=int, default=None,
                    help="Soft heap cap (supported on current impl); "
                         "on overflow return best-so-far")
    ap.add_argument("--verbose", action="store_true",
                    help="Stream BB events (opt_Z updates, prune) to stderr")
    args = ap.parse_args()

    module_name = IMPL_MODULES.get(args.impl, args.impl)
    hs = importlib.import_module(module_name)

    proc = psutil.Process()
    peak_rss = [0]
    cap_bytes = args.mem_cap_mb * 1024 * 1024
    model_ref = [None]

    def dump_partial(reason):
        m = model_ref[0]
        if m is None or m.opt_SP_tree is None:
            return
        partial = {
            "impl":          args.impl,
            "module":        module_name,
            "N":             args.N,
            "params":        dict(phi=args.phi, rho=args.rho, t=args.t,
                                  nu=args.nu, d=args.d, N=args.N),
            "opt_Z":         float(m.opt_Z),
            "SP_tree":       np.asarray(m.opt_SP_tree).astype(int).tolist(),
            "flow":          np.asarray(m.opt_flow).astype(float).tolist(),
            "seconds":       float("nan"),
            "peak_rss_mb":   peak_rss[0] / 1024 / 1024,
            "partial":       True,
            "reason":        reason,
        }
        with open(args.out_path, "w") as f:
            json.dump(partial, f)

    def check_mem(tag=""):
        rss = proc.memory_info().rss
        if rss > peak_rss[0]:
            peak_rss[0] = rss
        if rss > cap_bytes:
            print(
                f"!! MEMORY CAP EXCEEDED ({tag}): "
                f"rss={rss/1024/1024:.1f}MB > {args.mem_cap_mb}MB",
                file=sys.stderr, flush=True,
            )
            dump_partial(f"mem_cap at {rss/1024/1024:.1f}MB")
            os._exit(99)

    # Hook every leaf check: monitor mid-solve memory AND, when verbose,
    # stream opt_Z improvements to stderr.
    # v1 (paper-faithful) uses `check_opt`; v2 (improved) uses `_check_opt`
    # (with `check_opt` as alias). Hook whichever is available.
    orig_check = getattr(hs.BB_model, "_check_opt", None) or hs.BB_model.check_opt
    last_opt = [float("inf")]
    def patched_check(self, leaf):
        check_mem("check_opt")
        if args.verbose and leaf.Z < last_opt[0]:
            last_opt[0] = float(leaf.Z)
            print(f"  [opt_Z={leaf.Z:.6f}  rss={proc.memory_info().rss/1024/1024:.0f}MB]",
                  file=sys.stderr, flush=True)
        return orig_check(self, leaf)
    if hasattr(hs.BB_model, "_check_opt"):
        hs.BB_model._check_opt = patched_check
    hs.BB_model.check_opt  = patched_check

    params = dict(phi=args.phi, rho=args.rho, t=args.t,
                  nu=args.nu, d=args.d, N=args.N)
    prm = hs.Parameter(**params)
    net = hs.Network(prm)
    init_lower = np.zeros(prm.L)
    init_upper = np.ones(prm.L) * (prm.nu * (prm.N - 2))

    # v2 (improved) accepts max_leaves; v1 / older variants may not. Pass
    # only if supported.
    try:
        model = hs.BB_model(prm, net, max_leaves=args.max_leaves)
    except TypeError:
        model = hs.BB_model(prm, net)
    model_ref[0] = model

    sink = io.StringIO()
    t0 = time.perf_counter()
    if args.verbose:
        opt_Z, opt_SP_tree, opt_flow = model.solve(init_lower, init_upper)
    else:
        with redirect_stdout(sink):
            opt_Z, opt_SP_tree, opt_flow = model.solve(init_lower, init_upper)
    elapsed = time.perf_counter() - t0
    check_mem("done")

    result = {
        "impl":          args.impl,
        "module":        module_name,
        "N":             args.N,
        "params":        params,
        "opt_Z":         float(opt_Z),
        "SP_tree":       np.asarray(opt_SP_tree).astype(int).tolist(),
        "flow":          np.asarray(opt_flow).astype(float).tolist(),
        "seconds":       elapsed,
        "peak_rss_mb":   peak_rss[0] / 1024 / 1024,
        "partial":       False,
    }
    with open(args.out_path, "w") as f:
        json.dump(result, f)

    print(
        f"{args.impl} N={args.N}: opt_Z={opt_Z:.9f}  "
        f"time={elapsed:.3f}s  peak_rss={result['peak_rss_mb']:.1f}MB",
        flush=True,
    )


if __name__ == "__main__":
    main()
