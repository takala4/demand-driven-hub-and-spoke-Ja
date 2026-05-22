#!/usr/bin/env python
# -*- coding: utf-8 -*-
# viz_utils.py
# Paper-quality visualisation utilities for hub-and-spoke trees.
#
# Logic extracted from viz.ipynb so that arbitrary SP_tree outputs from
# v1 / v2 / v3 / v4 / v5 can be plotted with the same paper-style figures
# without modifying viz.ipynb itself.
#
# Public API:
#   calc_out_flow(prm, net, SP_tree)
#       -> array of size N giving aggregate flow through each node
#   link_plot(prm, net, SP_tree, save_path=None, ...)
#       -> matplotlib Figure with IT links as straight black arcs and MT
#          links as curved red arcs, optional outflow bars on each node;
#          saves to PDF/PNG/SVG if save_path is given.

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Aggregate flow on each node's outgoing link
# ---------------------------------------------------------------------------

def calc_out_flow(prm, net, SP_tree) -> np.ndarray:
    """Flow leaving each node along its outgoing edge (= ν · subtree size).

    The source node 0 has no outgoing edge; out_flow[0] = 0.
    """
    out_flow = np.zeros(prm.N, dtype=np.float64)
    for from_node in range(1, prm.N):
        cur = int(from_node)
        while cur != 0:
            out_flow[cur] += prm.nu
            cur = int(SP_tree[cur])
    return out_flow


# ---------------------------------------------------------------------------
# Paper-style tree plot
# ---------------------------------------------------------------------------

def link_plot(prm, net, SP_tree, *,
              save_path: Optional[str] = None,
              show_outflow: bool = True,
              figsize: tuple = (12, 3.5),
              spacing: float = 35.0,
              it_color: str = "black",
              mt_color: str = "#FF4B00",
              mt_curvature: float = -0.3,
              line_width: float = 2.0,
              node_size: int = 500,
              font_family: str = "Times New Roman",
              font_size: int = 15,
              bar_color: str = "black",
              bar_width: float = 10.0,
              bar_max_height: float = 3.5,
              bar_min_height: float = 0.25,
              ymin: float = -2.0,
              ymax: Optional[float] = None,
              dpi: int = 400):
    """Paper-quality plot of a hub-and-spoke arborescence.

    Parameters
    ----------
    prm, net : Parameter / Network as produced by any hubspoke* module.
    SP_tree  : array, SP_tree[i] = parent of i (SP_tree[0] should be -1).
    save_path: if not None, save the figure to this path (format inferred
               from the extension; e.g. ``image3/HS_d.pdf``).
    show_outflow: if True, draw black bars above each node whose height
               reflects ν · (subtree size) — useful to visualise consolidation.
    spacing  : horizontal distance between consecutive nodes in the layout.
    it_color, mt_color, mt_curvature : edge styling.
    figsize, line_width, node_size, font_*, bar_* : matplotlib styling.
    dpi      : resolution when saving.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object (also implicitly the current figure).
    """
    G_IT = nx.DiGraph()
    G_MT = nx.DiGraph()

    pos = {i: (i * spacing, 0.0) for i in range(prm.N)}

    for i in range(1, prm.N):
        j = int(SP_tree[i])
        if j < 0:
            continue
        if abs(i - j) == 1:
            G_IT.add_edge(i, j)
        else:
            G_MT.add_edge(i, j)

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    for spine in ("right", "top", "bottom", "left"):
        ax.spines[spine].set_visible(False)
    ax.set_xlim(-12.0, (prm.N - 1) * spacing + 12.0)
    # ylim will be set after computing bar heights so bars are never clipped.
    ax.set_xticks([])
    ax.set_yticks([])

    common = dict(
        pos=pos,
        node_size=node_size,
        width=line_width,
        node_color="white",
        with_labels=True,
        font_family=font_family,
        font_size=font_size,
        edgecolors="black",
    )
    if G_IT.number_of_edges() > 0:
        nx.draw_networkx(G_IT, edge_color=it_color,
                         connectionstyle="arc3,rad=0.0", **common)
    if G_MT.number_of_edges() > 0:
        nx.draw_networkx(G_MT, edge_color=mt_color,
                         connectionstyle=f"arc3,rad={mt_curvature}", **common)
    # Ensure isolated source node 0 is still drawn (no incident edges in
    # G_IT/G_MT means networkx skipped it above).
    if prm.N > 0 and G_IT.number_of_edges() == 0 and G_MT.number_of_edges() == 0:
        G0 = nx.DiGraph()
        G0.add_nodes_from(range(prm.N))
        nx.draw_networkx(G0, **common)

    if show_outflow:
        out_flow = calc_out_flow(prm, net, SP_tree)
        max_flow = float(out_flow.max())
        if max_flow > 0:
            # Linear map from out_flow to [bar_min_height, bar_max_height].
            scale = (bar_max_height - bar_min_height) / max_flow
            bar_heights = bar_min_height + out_flow * scale
        else:
            bar_heights = np.full_like(out_flow, bar_min_height)
        bar_heights[0] = 0.0
        ax.bar([pos[i][0] for i in range(prm.N)],
               bar_heights, width=bar_width, color=bar_color)
        # Auto-fit ylim to include the tallest bar plus a small margin.
        top = max(bar_max_height + 0.5, float(bar_heights.max()) + 0.5)
    else:
        top = 4.0
    ax.set_ylim(ymin, ymax if ymax is not None else top)

    if save_path is not None:
        dirname = os.path.dirname(save_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)

    return fig
