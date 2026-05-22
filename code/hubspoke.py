#!/usr/bin/env python
# -*- coding: utf-8 -*-
# hubspoke.py
# Author: Takara Sakai
# Date: 2025-01-28
# Description: Demand-driven hub-and-spoke model (Soland-type branch-and-bound).
#
# This is v2 (the improved implementation). v1 (paper-faithful) lives in
# hubspoke_v1.py; both solve the same problem with the same formulation and
# agree on opt_Z and SP_tree for all paper cases.
#
# Public API (unchanged from v1):
#   Parameter(phi, rho, t, nu, d, N)
#   Network(prm)
#   Leaf(prm, net, flow_lower, flow_upper)
#   BB_model(prm, net).solve(init_flow_lower, init_flow_upper)
#       -> (opt_Z, opt_SP_tree, opt_flow)
#   link_plot(prm, net, SP_tree)
#   generate_flow(prm, net, SP_tree)
#
# Internal improvements over v1:
#   * Per-leaf solve is a single Numba-compiled kernel (`_solve_inner`) that
#     fuses vF / linearisation / dense Dijkstra / flow propagation / Z, app_Z
#     and the argmax over F - G into one pass with zero Python overhead.
#   * Network precomputes Numba-friendly arrays: a CSR-style (indptr, flat)
#     view of incoming links and a dense Node2Link matrix.
#   * BB_model adds correctness-preserving pruning (skip-on-push, periodic
#     heap rebuild, exact-leaf skip) plus a diff-based heap encoding with
#     chunked numpy history and struct-packed entries to keep memory bounded
#     even on the hardest paper case (d).

import numpy as np
import networkx as nx
import struct
from heapq import heappush, heappop, heapify
import matplotlib.pyplot as plt

from numba import njit

# Heap entries are packed as 28-byte big-endian bytes objects:
#   app_Z (float64) | k (uint64) | branch_link (int32) | branch_value (float64)
# Big-endian + non-negative app_Z makes lexicographic byte comparison match
# numeric (app_Z, k, branch_link, branch_value) order, so heapq still gives
# a min-heap by lower bound. Each entry is ~60 bytes incl. CPython overhead,
# vs ~200 bytes for the equivalent 4-tuple of Python objects.
_HEAP_FMT    = '>dQid'
_HEAP_STRUCT = struct.Struct(_HEAP_FMT)


# ---------------------------------------------------------------------------
# Numba kernel: full per-leaf solve
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _solve_inner(
    N, L, nu,
    flow_lower, flow_upper,
    is_IT, coef_IT, coef_MT,
    in_links_flat, in_links_indptr, link_from,
    node2link_mat,
):
    """Solve one Branch-and-Bound leaf in a single Numba pass.

    Returns SP_tree, flow, Z, app_Z, max_link_idx.
    """
    INF = 1.0e300

    # ----- vF on (upper, lower) + slope / intercept of the secant
    Fu        = np.empty(L, dtype=np.float64)
    Fl        = np.empty(L, dtype=np.float64)
    slope     = np.empty(L, dtype=np.float64)
    intercept = np.empty(L, dtype=np.float64)

    for i in range(L):
        u = flow_upper[i]
        l = flow_lower[i]
        if is_IT[i]:
            Fu[i] = coef_IT[i] * u
            Fl[i] = coef_IT[i] * l
        else:
            Fu[i] = coef_MT[i] * np.log(u + 1.0)
            Fl[i] = coef_MT[i] * np.log(l + 1.0)
        diff = u - l
        if diff == 0.0:
            slope[i]     = 0.0
            intercept[i] = 0.0
        else:
            slope[i]     = (Fu[i] - Fl[i]) / diff
            intercept[i] = (u * Fl[i] - l * Fu[i]) / diff

    # ----- Dense Dijkstra (O(N^2)); for tiny N this beats heap-based Dijkstra
    spl     = np.full(N, INF, dtype=np.float64)
    spl[0]  = 0.0
    visited = np.zeros(N, dtype=np.bool_)
    SP_tree = np.full(N, -1, dtype=np.int32)

    for _ in range(N):
        best_d = INF
        best_v = -1
        for v in range(N):
            if (not visited[v]) and spl[v] < best_d:
                best_d = spl[v]
                best_v = v
        if best_v < 0:
            break
        visited[best_v] = True
        start = in_links_indptr[best_v]
        end   = in_links_indptr[best_v + 1]
        for k in range(start, end):
            link_idx = in_links_flat[k]
            src = link_from[link_idx]
            if visited[src]:
                continue
            nc = best_d + slope[link_idx]
            if nc < spl[src]:
                spl[src]     = nc
                SP_tree[src] = best_v

    # ----- Generate flow by walking each node's path back to source 0
    flow = np.zeros(L, dtype=np.float64)
    for node_idx in range(1, N):
        cur = node_idx
        while cur != 0:
            parent = SP_tree[cur]
            link_idx = node2link_mat[cur, parent]
            flow[link_idx] += nu
            cur = parent

    # ----- Z, app_Z, and argmax(F - G)
    Z = 0.0
    app_Z = 0.0
    max_diff = -INF
    max_link_idx = 0
    for i in range(L):
        f = flow[i]
        if is_IT[i]:
            vF_i = coef_IT[i] * f
        else:
            vF_i = coef_MT[i] * np.log(f + 1.0)
        u = flow_upper[i]
        l = flow_lower[i]
        if u == l:
            vG_i = 0.0
        else:
            vG_i = f * slope[i] + intercept[i]
        Z     += vF_i
        app_Z += vG_i
        d = vF_i - vG_i
        if d > max_diff:
            max_diff = d
            max_link_idx = i

    return SP_tree, flow, Z, app_Z, max_link_idx


# ---------------------------------------------------------------------------
# Public classes
# ---------------------------------------------------------------------------

class Parameter:
    def __init__(self, phi, rho, t, nu, d, N):
        self.phi = phi
        self.rho = rho
        self.t   = t
        self.nu  = nu
        self.d   = d
        self.N   = N
        self.L   = (N - 1) * (N - 1)


class Network:
    def __init__(self, prm):
        self.prm = prm
        self._build()

    def _build(self):
        N = self.prm.N
        L = self.prm.L

        self.Node2Link = {}
        self.Link2Node = {}
        self.OutLinks  = {i: [] for i in range(1, N)}
        self.InLinks   = {i: [] for i in range(N)}

        self.distance   = np.zeros(L)
        self.IT_MT_bool = np.ones(L)

        link_from = np.empty(L, dtype=np.int32)
        link_to   = np.empty(L, dtype=np.int32)

        link_idx = 0
        for from_node in range(1, N):
            self.Node2Link[from_node] = {}
            for to_node in range(N):
                if from_node == to_node:
                    continue
                dist = abs(from_node - to_node)
                if dist == 1:
                    self.IT_MT_bool[link_idx] = 0.0
                self.distance[link_idx] = dist
                self.Node2Link[from_node][to_node] = link_idx
                self.Link2Node[link_idx] = (from_node, to_node)
                self.OutLinks[from_node].append(link_idx)
                self.InLinks[to_node].append(link_idx)
                link_from[link_idx] = from_node
                link_to[link_idx]   = to_node
                link_idx += 1

        self.link_from = link_from
        self.link_to   = link_to

        # CSR-style incoming-links view: in_links_flat[indptr[v]:indptr[v+1]]
        in_links_indptr = np.zeros(N + 1, dtype=np.int32)
        for v in range(N):
            in_links_indptr[v + 1] = in_links_indptr[v] + len(self.InLinks[v])
        in_links_flat = np.empty(in_links_indptr[N], dtype=np.int32)
        for v in range(N):
            for j, lk in enumerate(self.InLinks[v]):
                in_links_flat[in_links_indptr[v] + j] = lk
        self.in_links_flat   = in_links_flat
        self.in_links_indptr = in_links_indptr

        # Dense Node2Link matrix; -1 marks the missing diagonal / source outs
        node2link_mat = np.full((N, N), -1, dtype=np.int32)
        for from_node in range(1, N):
            for to_node, lk in self.Node2Link[from_node].items():
                node2link_mat[from_node, to_node] = lk
        self.node2link_mat = node2link_mat

        # IT / MT coefficients reused by every Leaf evaluation
        self.is_IT   = (self.distance == 1.0)
        self.coef_IT = self.prm.t * self.prm.d * self.distance
        self.coef_MT = self.prm.phi + self.prm.rho * self.prm.d * self.distance

    # Kept for backward compatibility with external callers.
    def make_data(self):
        return True


class Leaf:
    def __init__(self, prm, net, flow_lower, flow_upper):
        self.prm = prm
        self.net = net
        # Float64 arrays are required by the JIT kernel.
        self.flow_lower = np.ascontiguousarray(flow_lower, dtype=np.float64)
        self.flow_upper = np.ascontiguousarray(flow_upper, dtype=np.float64)

    def solve(self):
        SP_tree, flow, Z, app_Z, max_link_idx = _solve_inner(
            self.prm.N, self.prm.L, float(self.prm.nu),
            self.flow_lower, self.flow_upper,
            self.net.is_IT, self.net.coef_IT, self.net.coef_MT,
            self.net.in_links_flat, self.net.in_links_indptr, self.net.link_from,
            self.net.node2link_mat,
        )
        self.SP_tree      = SP_tree
        self.flow         = flow
        self.Z            = float(Z)
        self.app_Z        = float(app_Z)
        self.max_link_idx = int(max_link_idx)
        return True


class BB_model:
    # Diff-based heap encoding:
    #   * Heap entry  : (app_Z, k, my_branch_link, my_branch_value)
    #       — k is this node's id; (my_branch_link, my_branch_value) describe
    #         how *this* node's children will be branched.
    #   * History dict: _history[k] = (parent_k, edge_link, edge_value, edge_side)
    #       — describes how node k was created from its parent.
    #         edge_side: 0 = upper bound was tightened, 1 = lower bound was tightened.
    #         Root (k=0) has no edge and is not stored in _history.
    # At pop time we reconstruct (lower, upper) by walking the parent chain
    # from k back to the root, applying overrides. This collapses per-entry
    # memory from O(L) floats to a fixed handful of ints/floats.

    # History is stored as chunked numpy arrays — each new chunk is allocated
    # fresh (no resize-and-copy), so memory grows linearly with no 2x peaks.
    # Per entry overhead: 21 bytes (int64 + int32 + float64 + int8).
    _CHUNK_BITS = 20  # 1<<20 = ~1M entries per chunk
    _CHUNK_SIZE = 1 << _CHUNK_BITS
    _CHUNK_MASK = _CHUNK_SIZE - 1

    def __init__(self, prm, net, max_leaves=None, prune_every=10000):
        self.prm = prm
        self.net = net
        self.opt_Z = np.inf
        self.opt_SP_tree = None
        self.opt_flow    = None
        self.max_leaves  = max_leaves
        self.prune_every = prune_every
        self._history_size = 0
        self._h_parent_chunks = []
        self._h_link_chunks   = []
        self._h_value_chunks  = []
        self._h_side_chunks   = []
        self._init_lower  = None
        self._init_upper  = None

    def _record_history(self, k, parent_k, edge_link, edge_value, edge_side):
        chunk_idx = k >> self._CHUNK_BITS
        local_idx = k & self._CHUNK_MASK
        while chunk_idx >= len(self._h_parent_chunks):
            self._h_parent_chunks.append(np.empty(self._CHUNK_SIZE, dtype=np.int64))
            self._h_link_chunks.append(  np.empty(self._CHUNK_SIZE, dtype=np.int32))
            self._h_value_chunks.append( np.empty(self._CHUNK_SIZE, dtype=np.float64))
            self._h_side_chunks.append(  np.empty(self._CHUNK_SIZE, dtype=np.int8))
        self._h_parent_chunks[chunk_idx][local_idx] = parent_k
        self._h_link_chunks  [chunk_idx][local_idx] = edge_link
        self._h_value_chunks [chunk_idx][local_idx] = edge_value
        self._h_side_chunks  [chunk_idx][local_idx] = edge_side
        if k >= self._history_size:
            self._history_size = k + 1

    def _reconstruct(self, k):
        lower = self._init_lower.copy()
        upper = self._init_upper.copy()
        cur = k
        h_parent_chunks = self._h_parent_chunks
        h_link_chunks   = self._h_link_chunks
        h_value_chunks  = self._h_value_chunks
        h_side_chunks   = self._h_side_chunks
        CHUNK_BITS = self._CHUNK_BITS
        CHUNK_MASK = self._CHUNK_MASK
        while cur > 0:
            ci = cur >> CHUNK_BITS
            li = cur & CHUNK_MASK
            link = int(h_link_chunks[ci][li])
            val  = float(h_value_chunks[ci][li])
            if h_side_chunks[ci][li] == 0:
                if upper[link] > val:
                    upper[link] = val
            else:
                if lower[link] < val:
                    lower[link] = val
            cur = int(h_parent_chunks[ci][li])
        return lower, upper

    def _maybe_push(self, hq, leaf, k, parent_k, edge_link, edge_value, edge_side):
        if leaf.app_Z >= self.opt_Z:
            return False
        # Exact-leaf skip: if leaf.Z == leaf.app_Z, the secant linearisation
        # has zero gap and the leaf gives the box's true min. Branching can
        # only produce children with Z >= this leaf's Z, so it cannot improve
        # opt_Z further. Without this guard, degenerate boxes (e.g. all-IT
        # shortest paths where every F-G is 0) get branched on link 0 by the
        # argmax tie-break and the BB loops forever at a flat min_LB.
        if leaf.Z - leaf.app_Z < 1e-9:
            return False
        self._record_history(k, parent_k, edge_link, edge_value, edge_side)
        heappush(hq, _HEAP_STRUCT.pack(
            float(leaf.app_Z), k,
            int(leaf.max_link_idx),
            float(leaf.flow[leaf.max_link_idx]),
        ))
        return True

    def solve(self, init_flow_lower, init_flow_upper):
        k = 0
        iters_since_prune = 0
        opt_Z_at_last_prune = np.inf
        hit_cap = False

        self._init_lower = np.ascontiguousarray(init_flow_lower, dtype=np.float64)
        self._init_upper = np.ascontiguousarray(init_flow_upper, dtype=np.float64)
        self._history_size = 0

        init_leaf = Leaf(self.prm, self.net, self._init_lower, self._init_upper)
        init_leaf.solve()
        self._check_opt(init_leaf)

        hq = []
        # Root has no edge — push directly, do not record in history.
        if init_leaf.app_Z < self.opt_Z:
            heappush(hq, _HEAP_STRUCT.pack(
                float(init_leaf.app_Z), k,
                int(init_leaf.max_link_idx),
                float(init_leaf.flow[init_leaf.max_link_idx]),
            ))

        while hq:
            app_Z, popped_k, branch_link, branch_value = _HEAP_STRUCT.unpack(heappop(hq))

            if app_Z >= self.opt_Z:
                print('k=', popped_k)
                break

            # Reconstruct (lower, upper) for the popped node by walking the
            # parent chain. O(depth) lookups.
            lower, upper = self._reconstruct(popped_k)

            # Branch (1): tighten upper bound on branch_link to branch_value.
            k += 1
            new_upper = upper.copy()
            new_upper[branch_link] = branch_value
            # `lower` is local to this iteration; safe to share with child Leaf
            # (Leaf and _solve_inner only read it).
            child = Leaf(self.prm, self.net, lower, new_upper)
            child.solve()
            self._check_opt(child)
            self._maybe_push(hq, child, k, popped_k, branch_link, branch_value, 0)

            # Branch (2): tighten lower bound on branch_link to branch_value.
            k += 1
            new_lower = lower.copy()
            new_lower[branch_link] = branch_value
            child = Leaf(self.prm, self.net, new_lower, upper)
            child.solve()
            self._check_opt(child)
            self._maybe_push(hq, child, k, popped_k, branch_link, branch_value, 1)

            iters_since_prune += 1
            if iters_since_prune >= self.prune_every:
                improved = self.opt_Z < opt_Z_at_last_prune
                if improved:
                    # Bytes entries sort lex; the first 8 bytes encode app_Z
                    # as big-endian double. Build a sentinel for the same
                    # format and drop entries whose packed app_Z >= it.
                    threshold = _HEAP_STRUCT.pack(float(self.opt_Z), 0, 0, 0.0)[:8]
                    before = len(hq)
                    hq = [e for e in hq if e[:8] < threshold]
                    heapify(hq)
                    opt_Z_at_last_prune = self.opt_Z
                else:
                    before = len(hq)
                min_lb = _HEAP_STRUCT.unpack(hq[0])[0] if hq else self.opt_Z
                print(f'prune: {before} -> {len(hq)} (opt_Z={self.opt_Z}, '
                      f'min_LB={min_lb}, gap={self.opt_Z - min_lb:.4g}, '
                      f'hist={self._history_size})')
                iters_since_prune = 0

            if self.max_leaves is not None and len(hq) > self.max_leaves:
                hit_cap = True
                print(f'!! max_leaves cap hit ({len(hq)} > {self.max_leaves}); '
                      f'returning best-so-far opt_Z={self.opt_Z}')
                break

        self.hit_cap = hit_cap
        return self.opt_Z, self.opt_SP_tree, self.opt_flow

    def _check_opt(self, leaf):
        if leaf.Z < self.opt_Z:
            self.opt_Z       = leaf.Z
            self.opt_SP_tree = leaf.SP_tree
            self.opt_flow    = leaf.flow
            print('opt_Z=', self.opt_Z)
        return True

    # Backward-compatible alias
    check_opt = _check_opt


def generate_flow(prm, net, SP_tree):
    flow = np.zeros(prm.L)
    for node_idx in range(1, prm.N):
        cur = node_idx
        while cur != 0:
            parent = int(SP_tree[cur])
            flow[net.Node2Link[cur][parent]] += prm.nu
            cur = parent
    return flow


def link_plot(prm, net, SP_tree):
    G_IT_link = nx.DiGraph()
    G_MT_link = nx.DiGraph()

    pos = {i: (i * 35.0, 0.0) for i in range(prm.N)}

    for from_node_idx in range(1, prm.N):
        to_node_idx = int(SP_tree[from_node_idx])
        if abs(from_node_idx - to_node_idx) == 1:
            G_IT_link.add_edge(from_node_idx, to_node_idx)
        else:
            G_MT_link.add_edge(from_node_idx, to_node_idx)

    plt.figure(figsize=(12, 5))
    ax = plt.gca()
    for spine in ('right', 'top', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)

    draw_kwargs = dict(
        pos=pos,
        node_size=500,
        node_color='white',
        with_labels=True,
        font_family='Times New Roman',
        font_size=15,
        edgecolors='black',
    )
    nx.draw_networkx(G_IT_link, edge_color='black', connectionstyle='arc3,rad=0.0',
                     **draw_kwargs)
    nx.draw_networkx(G_MT_link, edge_color='red',   connectionstyle='arc3,rad=0.3',
                     **draw_kwargs)
    return True
