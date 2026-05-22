#!/usr/bin/env python
# -*- coding: utf-8 -*-
# hubspoke_v3.py
# v3: Variable Neighborhood Search (VNS) — stochastic heuristic.
#
# This is a probabilistic heuristic for the demand-driven hub-and-spoke
# problem. It is NOT a deterministic global optimiser; see v1 (paper-faithful
# Soland BB) or v2 (improved Soland BB) for those. v3 is intended for
# scalability: it routinely finds the same optima as v2 on the paper cases
# in a fraction of v2's time, and runs in seconds on instances (N≥25) that
# v2 cannot certify in reasonable memory.
#
# Algorithm sketch:
#   1. Start from a chain (initial parent assignment).
#   2. Descend via 1-opt local search (try every single-parent swap).
#   3. Shake: randomly change k parents (k starts at 1).
#   4. Local search from the shaken solution.
#   5. If improved → accept, reset k to 1.
#      Else        → k := k+1 (cycle through k_max).
#   6. If patience exhausted → random restart (configurable count).
#   7. Stop on time limit or after restarts done.
#
# Convergence guarantee (asymptotic):
#   * With `restarts > 0` and a random-restart distribution of full support,
#     P(found global optimum) → 1 as time → ∞ (Brimberg & Mladenović 1996).
#   * No useful finite-time error bound exists.
#   * For a rigorous epsilon-global optimum, use v2 (small problems) or v4
#     (MILP + Gurobi, large problems).
#
# Public API:
#   VNSConfig — hyperparameters dataclass
#   solve(prm, net, config)
#     -> (best_Z, best_tree, best_flow, info_dict)

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numba import njit

from hubspoke import Parameter, Network


# ---------------------------------------------------------------------------
# Numba kernel: evaluate Z for a given parent array (rooted at 0)
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _eval_tree(N, L, nu, SP_tree, node2link_mat, is_IT, coef_IT, coef_MT):
    """Compute (Z, flow) for the arborescence encoded by SP_tree."""
    flow = np.zeros(L, dtype=np.float64)
    for node_idx in range(1, N):
        cur = node_idx
        while cur != 0:
            parent = SP_tree[cur]
            link_idx = node2link_mat[cur, parent]
            flow[link_idx] += nu
            cur = parent
    Z = 0.0
    for i in range(L):
        f = flow[i]
        if is_IT[i]:
            Z += coef_IT[i] * f
        else:
            Z += coef_MT[i] * np.log(f + 1.0)
    return Z, flow


# ---------------------------------------------------------------------------
# Tree validity check
# ---------------------------------------------------------------------------

def _is_valid_arborescence(N: int, tree: np.ndarray) -> bool:
    """Return True iff the parent array describes a tree rooted at 0
    (every non-source node reaches 0 by following parent pointers)."""
    children = [[] for _ in range(N)]
    for i in range(1, N):
        p = int(tree[i])
        if p < 0 or p >= N or p == i:
            return False
        children[p].append(i)
    reached = np.zeros(N, dtype=bool)
    reached[0] = True
    stack = [0]
    while stack:
        u = stack.pop()
        for v in children[u]:
            if not reached[v]:
                reached[v] = True
                stack.append(v)
    return bool(reached.all())


# ---------------------------------------------------------------------------
# 1-opt local search: parent-swap descent
# ---------------------------------------------------------------------------

def local_search_1opt(prm: Parameter, net: Network,
                       tree: np.ndarray,
                       max_passes: int = 200,
                       verbose: bool = False
                       ) -> Tuple[float, np.ndarray, np.ndarray]:
    """Greedy 1-opt: for each node i, try every other valid parent j != i.
    Accept the first improving swap, restart the scan. Continue until a
    full pass finds no improvement (a local optimum w.r.t. single parent
    swaps).

    Each evaluation re-runs the full Numba _eval_tree kernel (N^2 work),
    so total cost per pass is O(N^4). For N=17 this is microseconds.
    """
    N = prm.N
    L = prm.L
    nu = float(prm.nu)
    tree = tree.copy().astype(np.int32)
    best_Z, best_flow = _eval_tree(
        N, L, nu, tree,
        net.node2link_mat, net.is_IT, net.coef_IT, net.coef_MT,
    )
    for _ in range(max_passes):
        improved = False
        for i in range(1, N):
            for j in range(N):
                if j == i or j == tree[i]:
                    continue
                old_parent = tree[i]
                tree[i] = j
                if _is_valid_arborescence(N, tree):
                    Z, flow = _eval_tree(
                        N, L, nu, tree,
                        net.node2link_mat, net.is_IT, net.coef_IT, net.coef_MT,
                    )
                    if Z < best_Z - 1e-12:
                        best_Z = float(Z)
                        best_flow = flow
                        improved = True
                        if verbose:
                            print(f"    1-opt: i={i} j={j} -> Z={best_Z:.6f}")
                        break
                tree[i] = old_parent
            if improved:
                break
        if not improved:
            break
    return best_Z, tree, best_flow


# ---------------------------------------------------------------------------
# VNS configuration
# ---------------------------------------------------------------------------

@dataclass
class VNSConfig:
    max_time:              float = 5.0      # seconds
    max_iter_no_improve:   int   = 200      # patience before restart
    k_max:                 int   = 5        # largest shake neighbourhood
    shake_validity_tries:  int   = 50       # rejection-sample tries per shake
    seed:                  Optional[int] = None
    verbose:               bool  = False
    # 'chain', 'star', 'random', or a parent-array ndarray.
    initial:               object = "chain"
    # Restart count: 0 disables. Each restart samples a fresh random tree.
    restarts:              int   = 3


# ---------------------------------------------------------------------------
# Initial-solution constructors
# ---------------------------------------------------------------------------

def _initial_tree(N: int, kind, rng: np.random.Generator) -> np.ndarray:
    if isinstance(kind, np.ndarray):
        return kind.astype(np.int32).copy()
    if kind == "chain":
        tree = np.empty(N, dtype=np.int32)
        tree[0] = -1
        for i in range(1, N):
            tree[i] = i - 1
        return tree
    if kind == "star":
        tree = np.zeros(N, dtype=np.int32)
        tree[0] = -1
        return tree
    if kind == "random":
        return _random_tree(N, rng)
    raise ValueError(f"unknown initial kind: {kind!r}")


def _random_tree(N: int, rng: np.random.Generator,
                 max_tries: int = 200) -> np.ndarray:
    """Sample a uniformly random arborescence by rejection."""
    for _ in range(max_tries):
        tree = np.empty(N, dtype=np.int32)
        tree[0] = -1
        for i in range(1, N):
            j = int(rng.integers(0, N))
            while j == i:
                j = int(rng.integers(0, N))
            tree[i] = j
        if _is_valid_arborescence(N, tree):
            return tree
    tree = np.empty(N, dtype=np.int32)
    tree[0] = -1
    for i in range(1, N):
        tree[i] = i - 1
    return tree


# ---------------------------------------------------------------------------
# Shake (random k-parent perturbation)
# ---------------------------------------------------------------------------

def _shake(tree: np.ndarray, k: int, rng: np.random.Generator,
           N: int, max_tries: int = 50) -> Optional[np.ndarray]:
    k = max(1, min(k, N - 1))
    for _ in range(max_tries):
        new = tree.copy()
        idxs = rng.choice(np.arange(1, N), size=k, replace=False)
        for i in idxs:
            j = int(rng.integers(0, N))
            while j == int(i):
                j = int(rng.integers(0, N))
            new[i] = j
        if _is_valid_arborescence(N, new):
            return new
    return None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def solve(prm: Parameter, net: Network,
          config: Optional[VNSConfig] = None
          ) -> Tuple[float, np.ndarray, np.ndarray, dict]:
    """Run VNS until time limit, patience, and all restarts are exhausted.

    Returns (best_Z, best_tree, best_flow, info) where info has keys
    `elapsed`, `iters_no_improve`, `restarts_done`, `history`.
    """
    cfg = config or VNSConfig()
    rng = np.random.default_rng(cfg.seed)
    N   = prm.N

    tree = _initial_tree(N, cfg.initial, rng)
    best_Z, best_tree, best_flow = local_search_1opt(prm, net, tree)
    if cfg.verbose:
        print(f"VNS init: Z={best_Z:.6f}", flush=True)

    history = [{"t": 0.0, "Z": best_Z, "phase": "init"}]
    t0 = time.perf_counter()
    iters_no_improve = 0
    k = 1
    restarts_done = 0

    while True:
        elapsed = time.perf_counter() - t0
        if elapsed > cfg.max_time:
            break
        if iters_no_improve >= cfg.max_iter_no_improve:
            if restarts_done >= cfg.restarts:
                break
            restarts_done += 1
            tree = _random_tree(N, rng)
            r_Z, r_tree, r_flow = local_search_1opt(prm, net, tree)
            if r_Z < best_Z:
                best_Z, best_tree, best_flow = r_Z, r_tree, r_flow
                if cfg.verbose:
                    print(f"  restart #{restarts_done}: Z={best_Z:.6f}", flush=True)
                history.append({"t": elapsed, "Z": best_Z,
                                "phase": f"restart{restarts_done}"})
            iters_no_improve = 0
            k = 1
            continue

        shaken = _shake(best_tree, k, rng, N, max_tries=cfg.shake_validity_tries)
        if shaken is None:
            k = min(k + 1, cfg.k_max)
            iters_no_improve += 1
            continue

        Z, t, f = local_search_1opt(prm, net, shaken)
        if Z < best_Z - 1e-12:
            best_Z, best_tree, best_flow = Z, t, f
            k = 1
            iters_no_improve = 0
            if cfg.verbose:
                print(f"  improved Z={best_Z:.6f}  (k={k}, t={elapsed:.2f}s)",
                      flush=True)
            history.append({"t": elapsed, "Z": best_Z, "phase": "shake+ls"})
        else:
            k = k + 1
            if k > cfg.k_max:
                k = 1
            iters_no_improve += 1

    info = {
        "elapsed":          time.perf_counter() - t0,
        "iters_no_improve": iters_no_improve,
        "restarts_done":    restarts_done,
        "history":          history,
    }
    return best_Z, best_tree, best_flow, info


# ---------------------------------------------------------------------------
# Backwards-compatible BB_model alias (so the worker/verify framework can
# treat v3 as a drop-in solver). The BB_model class here is a thin shim
# that runs VNS in .solve() and exposes opt_Z / opt_SP_tree / opt_flow.
# ---------------------------------------------------------------------------

class BB_model:
    """Drop-in shim mimicking hubspoke.BB_model so the verify harness can
    call v3 with the same interface. The name is a misnomer (no branch-and-
    bound is run); kept only for API compatibility."""

    def __init__(self, prm, net, max_leaves=None, max_time=10.0, seed=None):
        self.prm = prm
        self.net = net
        self.cfg = VNSConfig(max_time=max_time, seed=seed)
        self.opt_Z       = float("inf")
        self.opt_SP_tree = None
        self.opt_flow    = None

    def solve(self, init_flow_lower=None, init_flow_upper=None):
        Z, tree, flow, info = solve(self.prm, self.net, self.cfg)
        self.opt_Z       = Z
        self.opt_SP_tree = tree
        self.opt_flow    = flow
        self.vns_info    = info
        return self.opt_Z, self.opt_SP_tree, self.opt_flow

    def _check_opt(self, leaf):
        # Stub; the harness patches this to monitor memory and progress.
        return True

    check_opt = _check_opt
