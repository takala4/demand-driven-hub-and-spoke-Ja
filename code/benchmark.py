#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Benchmark v1 / v2 / v3 / v4 against each other.

Runs each requested case for each enabled implementation and prints
opt_Z, wall-clock time, and pairwise consistency with v1 (the
paper-faithful reference). v1 and case (d) are slow (~1 hour); pass
--no-v1 or --skip-d to skip them while iterating.

v2 / v4 produce deterministic answers; v3 is a stochastic heuristic and
the run is one realisation — the time budget is bounded by VNSConfig.

Usage:
    python benchmark.py                         # case (e), v1/v2 only
    python benchmark.py all                     # all 6 cases, v1/v2
    python benchmark.py all --no-v1             # skip the slow paper code
    python benchmark.py all --v3 --v4           # include heuristic + MILP
    python benchmark.py "(d)" --v3 --v4 --no-v1 # focused comparison
"""

from __future__ import annotations

import io
import sys
import time
from contextlib import redirect_stdout

import numpy as np

import hubspoke    as hs_v2  # improved BB (Numba)
import hubspoke_v1 as hs_v1  # paper-faithful BB

try:
    import hubspoke_v3 as hs_v3
    _HAS_V3 = True
except ImportError:
    _HAS_V3 = False

try:
    import hubspoke_v4 as hs_v4
    _HAS_V4 = True
except ImportError:
    _HAS_V4 = False


CASES = {
    "(a)": dict(phi=50.0, rho=20.0, t=1.0, nu=10.0, d=1.0, N=17),
    "(b)": dict(phi=50.0, rho=10.0, t=1.0, nu=10.0, d=1.0, N=17),
    "(c)": dict(phi=10.0, rho=10.0, t=1.0, nu=10.0, d=1.0, N=17),
    "(d)": dict(phi=10.0, rho=5.0,  t=1.0, nu=10.0, d=1.0, N=17),
    "(e)": dict(phi=50.0, rho=2.5,  t=1.0, nu=10.0, d=1.0, N=17),
    "(f)": dict(phi=25.0, rho=2.5,  t=1.0, nu=10.0, d=1.0, N=17),
}


def _setup(prm, module):
    net = module.Network(prm)
    init_lower = np.zeros(prm.L)
    init_upper = np.ones(prm.L) * (prm.nu * (prm.N - 2))
    return net, init_lower, init_upper


def run_bb(module, params):
    """For modules whose solve signature is .BB_model(prm, net).solve(l, u)."""
    prm = module.Parameter(**params)
    net, lo, up = _setup(prm, module)
    model = module.BB_model(prm, net)
    buf = io.StringIO()
    t0 = time.perf_counter()
    with redirect_stdout(buf):
        Z, tree, flow = model.solve(lo, up)
    return {"Z": float(Z), "SP_tree": np.asarray(tree), "flow": np.asarray(flow),
            "seconds": time.perf_counter() - t0}


def run_vns(params, max_time=15.0, seed=42, restarts=3):
    prm = hs_v3.Parameter(**params)
    net = hs_v3.Network(prm)
    cfg = hs_v3.VNSConfig(max_time=max_time, seed=seed, restarts=restarts,
                          max_iter_no_improve=300)
    t0 = time.perf_counter()
    Z, tree, flow, _ = hs_v3.solve(prm, net, cfg)
    return {"Z": float(Z), "SP_tree": np.asarray(tree), "flow": np.asarray(flow),
            "seconds": time.perf_counter() - t0}


def run_milp(params, time_limit=120.0, warm_tree=None):
    prm = hs_v4.Parameter(**params)
    net = hs_v4.Network(prm)
    cfg = hs_v4.V4Config(time_limit=time_limit, mip_gap=1e-6, mip_focus=3,
                          verbose=False)
    t0 = time.perf_counter()
    Z, tree, flow, info = hs_v4.solve(prm, net, cfg, warm_start_tree=warm_tree)
    return {"Z": float(Z), "SP_tree": np.asarray(tree), "flow": np.asarray(flow),
            "seconds": time.perf_counter() - t0, "info": info}


def warmup_v2():
    """Trigger Numba compile so the JIT cost is not charged to the first case."""
    params = dict(phi=50.0, rho=20.0, t=1.0, nu=10.0, d=1.0, N=4)
    t0 = time.perf_counter()
    _ = run_bb(hs_v2, params)
    return time.perf_counter() - t0


def compare(label, params, *, run_v1, run_v3_flag, run_v4_flag):
    print(f"=== Case {label}: {params} ===")
    results = {}

    if run_v1:
        print("  v1 ...", flush=True)
        results["v1"] = run_bb(hs_v1, params)
        r = results["v1"]
        print(f"  v1 : Z = {r['Z']:.6f}  time = {r['seconds']:.3f}s")

    print("  v2 ...", flush=True)
    results["v2"] = run_bb(hs_v2, params)
    r = results["v2"]
    print(f"  v2 : Z = {r['Z']:.6f}  time = {r['seconds']:.3f}s")

    if run_v3_flag and _HAS_V3:
        print("  v3 ...", flush=True)
        results["v3"] = run_vns(params)
        r = results["v3"]
        print(f"  v3 : Z = {r['Z']:.6f}  time = {r['seconds']:.3f}s  (heuristic)")

    if run_v4_flag and _HAS_V4:
        print("  v4 ...", flush=True)
        warm = results["v3"]["SP_tree"] if run_v3_flag else None
        results["v4"] = run_milp(params, warm_tree=warm)
        r = results["v4"]
        gap = r["info"]["mip_gap"]
        certified = "exact" if gap < 1e-6 else f"gap={gap:.1e}"
        print(f"  v4 : Z = {r['Z']:.6f}  time = {r['seconds']:.3f}s  ({certified})")

    # Consistency check against v1 (or v2 if no v1).
    ref_key = "v1" if "v1" in results else "v2"
    ref = results[ref_key]
    for key, val in results.items():
        if key == ref_key:
            continue
        z_ok    = abs(val["Z"] - ref["Z"]) < 1e-6 * max(abs(ref["Z"]), 1.0)
        tree_ok = np.array_equal(val["SP_tree"], ref["SP_tree"])
        flag = "OK" if (z_ok and tree_ok) else f"z_ok={z_ok} tree_ok={tree_ok}"
        print(f"  {key} vs {ref_key}: {flag}")

    print()
    return {"label": label, **results}


def main():
    argv = sys.argv[1:]
    skip_v1 = "--no-v1" in argv
    run_v3  = "--v3" in argv or "--all-impls" in argv
    run_v4  = "--v4" in argv or "--all-impls" in argv
    argv = [a for a in argv if not a.startswith("--")]

    labels = argv or ["(e)"]
    if labels == ["all"]:
        labels = list(CASES)

    print(f"warming up Numba JIT ...", flush=True)
    warm = warmup_v2()
    print(f"  JIT warm-up: {warm:.3f}s\n")

    results = []
    for lbl in labels:
        if lbl not in CASES:
            raise SystemExit(f"unknown case {lbl}; choose from {list(CASES)} or 'all'")
        results.append(compare(lbl, CASES[lbl],
                               run_v1=(not skip_v1),
                               run_v3_flag=run_v3,
                               run_v4_flag=run_v4))

    print("==== summary ====")
    cols = ["case"]
    if not skip_v1: cols.append("v1 (s)")
    cols.append("v2 (s)")
    if run_v3 and _HAS_V3: cols.append("v3 (s)")
    if run_v4 and _HAS_V4: cols.append("v4 (s)")
    cols.append("v2 opt_Z")
    print(f"{cols[0]:<6}", end="")
    for h in cols[1:]:
        print(f" {h:>12}", end="")
    print()
    for r in results:
        print(f"{r['label']:<6}", end="")
        if not skip_v1: print(f" {r['v1']['seconds']:>12.3f}", end="")
        print(f" {r['v2']['seconds']:>12.3f}", end="")
        if run_v3 and _HAS_V3: print(f" {r['v3']['seconds']:>12.3f}", end="")
        if run_v4 and _HAS_V4: print(f" {r['v4']['seconds']:>12.3f}", end="")
        print(f" {r['v2']['Z']:>12.6f}")


if __name__ == "__main__":
    main()
