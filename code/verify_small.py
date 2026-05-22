#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Quick correctness check on small problems with subprocess isolation.

Each implementation runs in its own subprocess with a hard RSS cap; if a
variant misbehaves the orchestrator survives and partial results are
preserved on disk.

Default comparison is `v2` (hubspoke.py, improved) vs `v1` (hubspoke_v1.py,
paper-faithful). When adding a new variant (e.g. a logic-level alternative
in `hubspoke_<variant>.py`), pass the module name to --impl directly:

    python verify_small.py 17 --impl hubspoke_my_variant

Usage:
    python verify_small.py 5                  # N=5, v2 vs v1
    python verify_small.py 8 --impl v2        # only the improved implementation
    python verify_small.py 10 --mem-cap-mb 4096 --timeout 600
    python verify_small.py 5 --phi 50 --rho 2.5   # override case parameters
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parent
WORKER   = CODE_DIR / "_verify_worker.py"


def run_worker(impl, N, out_path, mem_cap_mb, timeout_s, case_overrides):
    cmd = [sys.executable, str(WORKER), impl, str(N), str(out_path), str(mem_cap_mb)]
    for k, v in case_overrides.items():
        if v is None:
            cmd += [f"--{k}"]
        else:
            cmd += [f"--{k}", str(v)]

    print(f"  -> {impl} (N={N}, cap={mem_cap_mb}MB, timeout={timeout_s}s)", flush=True)
    try:
        r = subprocess.run(cmd, cwd=str(CODE_DIR),
                           timeout=timeout_s, capture_output=True, text=True)
    except subprocess.TimeoutExpired as exc:
        err = exc.stderr
        if err:
            if isinstance(err, bytes):
                err = err.decode("utf-8", "replace")
            for line in err.splitlines()[-30:]:
                print("    [stderr] " + line)
        return {"status": "timeout"}

    if r.stderr.strip():
        for line in r.stderr.strip().splitlines():
            print("    [stderr] " + line)
    if r.returncode == 99:
        return {"status": "mem_cap",  "stderr": r.stderr.strip()}
    if r.returncode != 0:
        return {"status": "error",
                "code":   r.returncode,
                "stdout": r.stdout.strip(),
                "stderr": r.stderr.strip()}

    with open(out_path) as f:
        data = json.load(f)
    if r.stdout.strip():
        print("    " + r.stdout.strip())
    return {"status": "ok", **data}


def compare(a, b, label_a, label_b):
    z_diff   = a["opt_Z"] - b["opt_Z"]
    z_ok     = abs(z_diff) < 1e-9
    tree_ok  = a["SP_tree"] == b["SP_tree"]

    fa = np.asarray(a["flow"])
    fb = np.asarray(b["flow"])
    flow_ok = bool(np.allclose(fa, fb, rtol=1e-9, atol=1e-9))

    print()
    print(f"  opt_Z     : {label_a}={a['opt_Z']:.9f}  {label_b}={b['opt_Z']:.9f}  "
          f"diff={z_diff:+.2e}  {'OK' if z_ok else 'MISMATCH'}")
    print(f"  SP_tree   : {'OK' if tree_ok else 'MISMATCH'}")
    if not tree_ok:
        for i, (x, y) in enumerate(zip(a["SP_tree"], b["SP_tree"])):
            if x != y:
                print(f"      node {i}: {label_a}={x}  {label_b}={y}")
    print(f"  flow      : {'allclose' if flow_ok else 'MISMATCH'}"
          f"   (max|diff|={float(np.max(np.abs(fa-fb))):.2e})")
    return z_ok and tree_ok and flow_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("N", type=int)
    ap.add_argument("--impl", default="both",
                    help="'v2' (hubspoke, improved), 'v1' (hubspoke_v1, paper-faithful), "
                         "'both' for comparison, or any importable module name.")
    ap.add_argument("--baseline", default="v1",
                    help="Baseline for --impl=both comparison (default: v1)")
    ap.add_argument("--mem-cap-mb", type=float, default=2048.0)
    ap.add_argument("--timeout",    type=float, default=300.0)
    ap.add_argument("--phi", type=float, default=None)
    ap.add_argument("--rho", type=float, default=None)
    ap.add_argument("--t",   type=float, default=None)
    ap.add_argument("--nu",  type=float, default=None)
    ap.add_argument("--d",   type=float, default=None)
    ap.add_argument("--max-leaves", type=int, default=None,
                    help="Soft heap cap (returns best-so-far on overflow)")
    ap.add_argument("--verbose", action="store_true",
                    help="Stream BB progress to stderr")
    args = ap.parse_args()

    overrides = {k: getattr(args, k) for k in ("phi", "rho", "t", "nu", "d")
                 if getattr(args, k) is not None}
    if args.max_leaves is not None:
        overrides["max-leaves"] = args.max_leaves
    if args.verbose:
        overrides["verbose"] = None

    if args.impl == "both":
        impls = [args.baseline, "v2"]
    else:
        impls = [args.impl]

    print(f"=== Verify N={args.N}  overrides={overrides or '(case-a defaults)'} ===")
    results = {}
    for impl in impls:
        out = CODE_DIR / f"_verify_{impl}_N{args.N}.json"
        r = run_worker(impl, args.N, out, args.mem_cap_mb, args.timeout, overrides)
        results[impl] = r
        if r["status"] != "ok":
            print(f"  !! {impl} status={r['status']}")
            for k, v in r.items():
                if k != "status" and v:
                    print(f"     {k}: {v}")
            sys.exit(2)

    if len(impls) == 2:
        ok = compare(results[impls[0]], results[impls[1]], impls[0], impls[1])
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
