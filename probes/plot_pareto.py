"""Figure F2: the resource-quality Pareto, two resolvable bands of resource variation.

For each instance the cheapest embedding is the origin, so x is the extra qubits an option
costs over the cheapest option for that same instance and y is the solve probability that
buys, relative to the instance mean. Under a resource-quality tradeoff both panels rise to
the right. Both use the same units on both axes, so the panels are directly comparable.

The third population in the data file, qubits absorbed into random chains, is not drawn: its
between-embedding spread is at or below the standard error of its own measurements, so it
bounds an effect rather than showing one. The bound is in the JSON.

  python3 probes/plot_pareto.py
"""

import json
import statistics as st
import sys

sys.path.insert(0, "/Users/nguyencongt/research-os/scripts")
import plot_utils                                       # noqa: E402
import matplotlib.pyplot as plt                         # noqa: E402
import numpy as np                                      # noqa: E402

SRC = "docs/paper/results/f2_pareto_resource_quality.json"
OUT = "docs/paper/figures/f2_pareto_resource_quality"


def centred(points, key_q, key_v):
    """Per instance: extra qubits over the cheapest option, quality minus the instance mean."""
    by = {}
    for p in points:
        by.setdefault(p["task"], []).append(p)
    xs, ys, tags = [], [], []
    for v in by.values():
        qmin = min(p[key_q] for p in v)
        vmean = st.mean(p[key_v] for p in v)
        for p in v:
            xs.append(p[key_q] - qmin)
            ys.append(p[key_v] - vmean)
            tags.append(p.get("arm"))
    return np.array(xs, float), np.array(ys, float), tags


def binned(x, y, edges, seed=0):
    mids, mean, lo, hi = [], [], [], []
    rng = np.random.default_rng(seed)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x < b)
        if m.sum() < 8:
            continue
        v = y[m]
        boot = np.sort([rng.choice(v, len(v)).mean() for _ in range(2000)])
        mids.append(v.size and x[m].mean())
        mean.append(v.mean())
        lo.append(boot[50])
        hi.append(boot[-50])
    return np.array(mids), np.array(mean), np.array(lo), np.array(hi)


def main():
    width = plot_utils.use_venue("neurips", "double", base=8)
    data = json.load(open(SRC))
    pops = {p["id"]: p for p in data["populations"]}
    c = plot_utils.OKABE_ITO

    fig, axes = plt.subplots(1, 2, figsize=(width, width * 0.33), sharey=True)

    # Panel A: the router's own sampling noise, a two-qubit band.
    p = pops["router_draws"]
    x, y, _ = centred(p["measurements"], "qubits", "p_solve")
    ax = axes[0]
    ax.scatter(x + np.random.default_rng(1).normal(0, 0.07, len(x)), y, s=5, alpha=0.28,
               color=c[0], linewidths=0, rasterized=True)
    mx, mm, lo, hi = binned(x, y, np.arange(-0.5, 8.5, 1.0), seed=1)
    ax.fill_between(mx, lo, hi, color=c[5], alpha=0.25, lw=0, zorder=4)
    ax.plot(mx, mm, color=c[5], marker="o", ms=3.5, lw=1.3, zorder=5)
    k = p["rho_qubits_vs_p_solve"]
    ax.set_title("A   one router, eight draws", pad=4)
    ax.set_ylabel("solve probability\nminus the instance mean")
    ax.set_xlabel("extra qubits over the cheapest")
    ax.text(0.96, 0.06, r"$\rho = %+.3f$  [%+.2f, %+.2f]" % (k["mean"], *k["ci95"]),
            transform=ax.transAxes, ha="right", va="bottom")
    ax.text(0.96, 0.96, "%d draws, %d instances" % (p["points"], p["instances"]),
            transform=ax.transAxes, ha="right", va="top", color="0.45")

    # Panel C: a change of method, a thirty-qubit band.
    p = pops["across_methods"]
    x, y, tags = centred(p["measurements"], "qubits", "p_solve")
    ax = axes[1]
    colour = {"minorminer": c[2], "pruned": c[1], "policy": c[4]}
    label = {"minorminer": "router", "pruned": "policy, pruned", "policy": "learned policy"}
    xs, ys = {}, {}
    for arm in ("minorminer", "pruned", "policy"):
        m = np.array([t == arm for t in tags])
        ax.scatter(x[m], y[m], s=6, alpha=0.22, color=colour[arm], linewidths=0, rasterized=True)
        xs[arm], ys[arm] = x[m].mean(), y[m].mean()
    ax.annotate("", xy=(xs["pruned"], ys["pruned"]), xytext=(xs["policy"], ys["policy"]),
                arrowprops=dict(arrowstyle="-|>", lw=1.1, color="0.3", shrinkA=7, shrinkB=7))
    for arm in ("minorminer", "pruned", "policy"):
        m = np.array([t == arm for t in tags])
        ax.errorbar(xs[arm], ys[arm], xerr=x[m].std(ddof=1) / m.sum() ** 0.5,
                    yerr=y[m].std(ddof=1) / m.sum() ** 0.5, color=colour[arm], marker="o",
                    ms=7, mec="white", mew=0.9, lw=1.3, zorder=6, label=label[arm])
    d = p["paired_differences"]["pruned_minus_policy"]
    ax.annotate("%.1f qubits deleted,\nquality %+.3f [%+.3f, %+.3f]"
                % (-d["qubits"]["mean"], d["p_solve"]["mean"], *d["p_solve"]["ci95"]),
                xy=((xs["policy"] + xs["pruned"]) / 2, ys["policy"]),
                xytext=(0, 20), textcoords="offset points", ha="center", va="bottom",
                color="0.25")
    ax.set_title("B   one instance, three constructions", pad=4)
    ax.set_xlabel("extra qubits over the cheapest")
    ax.legend(loc="lower left", handletextpad=0.4, borderaxespad=0.3)
    ax.text(0.96, 0.96, "%d embeddings, %d instances" % (p["points"], p["instances"]),
            transform=ax.transAxes, ha="right", va="top", color="0.45")

    for ax in axes:
        ax.axhline(0, color="0.55", lw=0.6, zorder=1)
        ax.margins(x=0.05)
    axes[0].set_ylim(-0.45, 0.45)

    fig.tight_layout(pad=0.4, w_pad=1.0)
    for path in plot_utils.save(fig, OUT):
        print("wrote", path)


if __name__ == "__main__":
    main()
