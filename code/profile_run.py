#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""cProfile the current (v1) hubspoke implementation on a chosen case."""

from __future__ import annotations

import cProfile
import io
import pstats
import sys
from contextlib import redirect_stdout

import numpy as np

import hubspoke as hs


CASES = {
    "(a)": dict(phi=50.0, rho=20.0, t=1.0, nu=10.0, d=1.0, N=17),
    "(b)": dict(phi=50.0, rho=10.0, t=1.0, nu=10.0, d=1.0, N=17),
    "(c)": dict(phi=10.0, rho=10.0, t=1.0, nu=10.0, d=1.0, N=17),
    "(d)": dict(phi=10.0, rho=5.0,  t=1.0, nu=10.0, d=1.0, N=17),
    "(e)": dict(phi=50.0, rho=2.5,  t=1.0, nu=10.0, d=1.0, N=17),
    "(f)": dict(phi=25.0, rho=2.5,  t=1.0, nu=10.0, d=1.0, N=17),
}


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "(e)"
    params = CASES[label]
    print(f"profiling case {label}: {params}")

    prm = hs.Parameter(**params)
    net = hs.Network(prm)
    init_lower = np.zeros(prm.L)
    init_upper = np.ones(prm.L) * (prm.nu * (prm.N - 2))
    model = hs.BB_model(prm, net)

    sink = io.StringIO()
    pr = cProfile.Profile()
    pr.enable()
    with redirect_stdout(sink):
        opt_Z, opt_SP_tree, opt_flow = model.solve(init_lower, init_upper)
    pr.disable()

    print(f"opt_Z = {opt_Z}")

    stats = pstats.Stats(pr).strip_dirs().sort_stats("cumulative")
    print("\n== top 25 by cumulative time ==")
    stats.print_stats(25)

    stats.sort_stats("tottime")
    print("\n== top 25 by self time ==")
    stats.print_stats(25)


if __name__ == "__main__":
    main()
