#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Benchmark the improved implementation (v2) against the paper-faithful one (v1).

Runs each requested case once per implementation, prints opt_Z + wall-clock
time, and checks that opt_Z and SP_tree match. The v1 run is slow
(case (d) takes ~1 hour) so use --no-v1 when iterating.

v2 (`hubspoke.py`) uses Numba; the first call triggers JIT compilation.
A small warm-up call is done before timing so the JIT cost is not charged
to the first case.

Usage:
    python benchmark.py                  # only case (e), both impls
    python benchmark.py all              # all 6 cases, both impls
    python benchmark.py all --no-v1      # all 6, v2 only
    python benchmark.py "(a)" "(b)"      # specific cases
"""

from __future__ import annotations

import io
import sys
import time
from contextlib import redirect_stdout

import numpy as np

import hubspoke    as hs_v2   # improved (Numba + BB optimisations)
import hubspoke_v1 as hs_v1   # paper-faithful


CASES = {
    "(a)": dict(phi=50.0, rho=20.0, t=1.0, nu=10.0, d=1.0, N=17),
    "(b)": dict(phi=50.0, rho=10.0, t=1.0, nu=10.0, d=1.0, N=17),
    "(c)": dict(phi=10.0, rho=10.0, t=1.0, nu=10.0, d=1.0, N=17),
    "(d)": dict(phi=10.0, rho=5.0,  t=1.0, nu=10.0, d=1.0, N=17),
    "(e)": dict(phi=50.0, rho=2.5,  t=1.0, nu=10.0, d=1.0, N=17),
    "(f)": dict(phi=25.0, rho=2.5,  t=1.0, nu=10.0, d=1.0, N=17),
}


def run_once(module, params):
    prm = module.Parameter(**params)
    net = module.Network(prm)
    init_lower = np.zeros(prm.L)
    init_upper = np.ones(prm.L) * (prm.nu * (prm.N - 2))
    model = module.BB_model(prm, net)

    buf = io.StringIO()
    t0 = time.perf_counter()
    with redirect_stdout(buf):
        opt_Z, opt_SP_tree, opt_flow = model.solve(init_lower, init_upper)
    t1 = time.perf_counter()
    return {
        "opt_Z": opt_Z,
        "SP_tree": np.asarray(opt_SP_tree),
        "flow": np.asarray(opt_flow),
        "seconds": t1 - t0,
    }


def warmup_v2():
    # Trigger Numba compile so it isn't charged to the first measured case.
    params = dict(phi=50.0, rho=20.0, t=1.0, nu=10.0, d=1.0, N=4)
    t0 = time.perf_counter()
    _ = run_once(hs_v2, params)
    return time.perf_counter() - t0


def compare(label, params, *, skip_v1=False):
    print(f"=== Case {label}: {params} ===")

    if skip_v1:
        v1 = None
    else:
        print("  v1 ...", flush=True)
        v1 = run_once(hs_v1, params)
        print(f"  v1 : opt_Z = {v1['opt_Z']:.6f}  time = {v1['seconds']:.3f}s")

    print("  v2 ...", flush=True)
    v2 = run_once(hs_v2, params)
    print(f"  v2 : opt_Z = {v2['opt_Z']:.6f}  time = {v2['seconds']:.3f}s")

    if v1 is not None:
        z_ok    = np.isclose(v1["opt_Z"], v2["opt_Z"], rtol=1e-9, atol=1e-9)
        tree_ok = np.array_equal(v1["SP_tree"], v2["SP_tree"])
        print(f"  opt_Z eq    : {z_ok}")
        print(f"  SP_tree eq  : {tree_ok}")

    print()
    return {"label": label, "v1": v1, "v2": v2}


def main():
    argv = sys.argv[1:]
    skip_v1 = "--no-v1" in argv
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
        results.append(compare(lbl, CASES[lbl], skip_v1=skip_v1))

    print("==== summary ====")
    hdr = ["case"]
    if not skip_v1:
        hdr.append("v1 (s)")
    hdr += ["v2 (s)", "speedup", "opt_Z"]
    print(f"{hdr[0]:<6}", end="")
    for h in hdr[1:]:
        print(f" {h:>14}", end="")
    print()

    for r in results:
        t_v2 = r["v2"]["seconds"]
        print(f"{r['label']:<6}", end="")
        if not skip_v1:
            t_v1 = r['v1']['seconds']
            print(f" {t_v1:>14.3f}", end="")
            sp = t_v1 / t_v2 if t_v2 > 0 else float('inf')
            print(f" {t_v2:>14.3f} {'x'+format(sp,'.1f'):>14}", end="")
        else:
            print(f" {t_v2:>14.3f} {'-':>14}", end="")
        print(f" {r['v2']['opt_Z']:>14.6f}")


if __name__ == "__main__":
    main()
