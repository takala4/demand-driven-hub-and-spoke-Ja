#!/usr/bin/env python
# -*- coding: utf-8 -*-
# hubspoke_v4.py
# v4: Mixed-Integer Linear Programming with piecewise-linear (PWL) cost
# approximation, solved by Gurobi. Deterministic epsilon-global optimum.
#
# Why this works exactly (not just approximately) for our problem
# -----------------------------------------------------------------
# Every link's flow is an integer multiple of ν: it equals ν times the
# number of demand sinks in the subtree rooted at the link's tail. With
# N nodes there are at most N-1 distinct positive multiples (ν, 2ν, ...,
# (N-1)ν). Placing PWL breakpoints at exactly {0, ν, 2ν, ..., (N-1)ν}
# therefore makes the PWL agree with the true concave cost F on every
# feasible point. The MILP solution is exact for the original problem
# (modulo Gurobi's numerical tolerance), not merely epsilon-approximate.
#
# Variables
#   y[a] ∈ {0,1}    link a used in the tree
#   x[a] ≥ 0        flow on link a (= ν · |subtree below tail of a| when used)
#   c[a] ≥ 0        cost on link a (PWL of x[a] for MT links, linear for IT)
#
# Constraints
#   y-link / x-link: x[a] ≤ M · y[a], where M = ν · (N-1).
#   Tree:            Σ_{a out of i} y[a] = 1 for every non-source i.
#   Flow conservation:
#     non-source i:  Σ_{a into i} x[a] + ν = Σ_{a out of i} x[a]
#     source 0:      Σ_{a into 0} x[a] = (N-1) · ν
#   PWL cost:        c[a] = PWL_a(x[a]) via Gurobi `addGenConstrPWL`.
#
# Objective: minimise Σ c[a].
#
# Public API
#   V4Config            — hyperparameters dataclass
#   solve(prm, net, config) -> (opt_Z, opt_SP_tree, opt_flow, info)
#   BB_model            — drop-in shim so the verify/test harness can use v4

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
    _HAS_GUROBI = True
except ImportError:  # pragma: no cover
    _HAS_GUROBI = False

from hubspoke import Parameter, Network


@dataclass
class V4Config:
    # If None, breakpoints = {0, nu, 2nu, ..., (N-1)*nu} (exact). Override
    # only if you want a coarser PWL (e.g. to time-trade tightness).
    breakpoints:  Optional[np.ndarray] = None
    time_limit:   float = 600.0       # Gurobi TimeLimit (seconds)
    mip_gap:      float = 1e-9        # MIPGap parameter
    threads:      int   = 0           # 0 = Gurobi default (all cores)
    verbose:      bool  = False
    presolve:     int   = -1          # Gurobi Presolve: -1 default, 0 off
    cuts:         int   = -1          # Gurobi Cuts: -1 default
    mip_focus:    int   = 0           # 0 default, 1 feasibility, 2 optimality, 3 bound


def _default_breakpoints(prm: Parameter) -> np.ndarray:
    """Discrete breakpoints at every reachable integer multiple of nu."""
    return np.arange(0, prm.N) * prm.nu  # 0, nu, 2nu, ..., (N-1)*nu


def solve(prm: Parameter, net: Network,
          config: Optional[V4Config] = None,
          warm_start_tree: Optional[np.ndarray] = None,
          ) -> Tuple[float, np.ndarray, np.ndarray, dict]:
    """Solve the demand-driven hub-and-spoke MILP via Gurobi.

    Set `warm_start_tree` (a parent array, e.g. from v3 VNS) to provide an
    initial integer-feasible solution. Gurobi will start with this as an
    incumbent and concentrate on closing the LP / cut bound to prove
    optimality, which is typically much faster than searching from scratch.
    """
    if not _HAS_GUROBI:
        raise RuntimeError(
            "gurobipy is required for v4. Install Gurobi and add it to PYTHONPATH."
        )
    cfg = config or V4Config()
    N = prm.N
    L = prm.L
    nu = float(prm.nu)
    M_flow = nu * (N - 1)
    breakpoints = cfg.breakpoints if cfg.breakpoints is not None \
                  else _default_breakpoints(prm)

    m = gp.Model("hubspoke_v4")
    if not cfg.verbose:
        m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", cfg.time_limit)
    m.setParam("MIPGap",    cfg.mip_gap)
    m.setParam("Threads",   cfg.threads)
    m.setParam("Presolve",  cfg.presolve)
    m.setParam("Cuts",      cfg.cuts)
    m.setParam("MIPFocus",  cfg.mip_focus)

    y = m.addVars(L, vtype=GRB.BINARY,          name="y")
    x = m.addVars(L, lb=0.0,    ub=M_flow,      name="x")
    c = m.addVars(L, lb=0.0,                     name="c")

    # Link activation: x[a] = 0 unless y[a] = 1.
    for a in range(L):
        m.addConstr(x[a] <= M_flow * y[a], name=f"act_{a}")

    # Cost: linear on IT, PWL on MT.
    for a in range(L):
        if net.is_IT[a]:
            m.addConstr(c[a] == net.coef_IT[a] * x[a], name=f"cIT_{a}")
        else:
            xpts = list(breakpoints.astype(float))
            ypts = [net.coef_MT[a] * float(np.log(xp + 1.0)) for xp in xpts]
            m.addGenConstrPWL(x[a], c[a], xpts, ypts, name=f"cMT_{a}")

    # Tree constraint: each non-source has exactly one outgoing link.
    for from_node in range(1, N):
        outs = net.OutLinks[from_node]
        m.addConstr(gp.quicksum(y[a] for a in outs) == 1,
                    name=f"tree_{from_node}")

    # Flow conservation.
    for v in range(N):
        in_links = net.InLinks[v]
        in_flow  = gp.quicksum(x[a] for a in in_links)
        if v == 0:
            m.addConstr(in_flow == (N - 1) * nu, name="cons_src")
        else:
            out_flow = gp.quicksum(x[a] for a in net.OutLinks[v])
            m.addConstr(in_flow + nu == out_flow, name=f"cons_{v}")

    m.setObjective(gp.quicksum(c[a] for a in range(L)), GRB.MINIMIZE)

    # Warm start (MIPStart): seed Gurobi with a known-feasible tree so it
    # starts with this UB and focuses on closing the LP / cut bound.
    if warm_start_tree is not None:
        ws_tree = np.asarray(warm_start_tree, dtype=np.int32)
        # Compute integer-feasible y / x from the tree.
        y_ws = np.zeros(L, dtype=np.float64)
        x_ws = np.zeros(L, dtype=np.float64)
        # subtree size at i = number of demands that route through i (incl. i)
        # Walk each demand to source, accumulating flow on traversed links.
        for node_idx in range(1, N):
            cur = int(node_idx)
            while cur != 0:
                parent = int(ws_tree[cur])
                a = int(net.node2link_mat[cur, parent])
                y_ws[a]  = 1.0
                x_ws[a] += nu
                cur = parent
        for a in range(L):
            y[a].Start = float(y_ws[a])
            x[a].Start = float(x_ws[a])

    m.optimize()

    if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        raise RuntimeError(f"Gurobi exited with status {m.Status}")
    if m.SolCount == 0:
        raise RuntimeError("Gurobi finished without a feasible solution")

    # Reconstruct tree and flow from y* / x*.
    SP_tree = np.full(N, -1, dtype=np.int32)
    flow    = np.zeros(L,  dtype=np.float64)
    for a in range(L):
        if y[a].X > 0.5:
            i, j = net.Link2Node[a]
            SP_tree[i] = j
            flow[a]    = x[a].X

    # Compute the exact objective with the true concave F (no PWL).
    Z_exact = 0.0
    for a in range(L):
        f = flow[a]
        if net.is_IT[a]:
            Z_exact += net.coef_IT[a] * f
        else:
            Z_exact += net.coef_MT[a] * np.log(f + 1.0)

    info = {
        "milp_obj":  float(m.ObjVal),    # PWL objective (= F at breakpoints)
        "mip_gap":   float(m.MIPGap),
        "runtime":   float(m.Runtime),
        "status":    int(m.Status),
        "num_pieces": int(len(breakpoints) - 1),
        "Z_exact":   float(Z_exact),
    }
    return Z_exact, SP_tree, flow, info


# ---------------------------------------------------------------------------
# Backwards-compatible BB_model shim (so the verify harness can drive v4
# without special-casing). The name is a misnomer; no branch-and-bound is
# implemented here — Gurobi handles all of that internally.
# ---------------------------------------------------------------------------

class BB_model:
    def __init__(self, prm, net, max_leaves=None,
                 time_limit: float = 600.0, mip_gap: float = 1e-9,
                 verbose: bool = False):
        self.prm = prm
        self.net = net
        self.cfg = V4Config(time_limit=time_limit, mip_gap=mip_gap,
                            verbose=verbose)
        self.opt_Z       = float("inf")
        self.opt_SP_tree = None
        self.opt_flow    = None

    def solve(self, init_flow_lower=None, init_flow_upper=None):
        Z, tree, flow, info = solve(self.prm, self.net, self.cfg)
        self.opt_Z       = Z
        self.opt_SP_tree = tree
        self.opt_flow    = flow
        self.gurobi_info = info
        return self.opt_Z, self.opt_SP_tree, self.opt_flow

    def _check_opt(self, leaf):
        return True

    check_opt = _check_opt
