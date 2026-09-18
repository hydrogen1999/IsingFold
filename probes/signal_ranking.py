"""How well each candidate selection signal ranks the embeddings a router actually produces.

The paper's thesis has been stated as "qubit count is not a quality signal". That is too strong
and a reviewer who computes the correlation will find it is wrong: qubit count does rank
candidates above chance. What the record supports is a comparison of strengths, on one scale.

This reads the selection-rule logs, which already hold, per instance, the qubit count and chain
length of every draw, a 256-read selection block, and a disjoint 512-read assessment. For each
candidate signal it computes the within-instance Spearman correlation against the assessment,
which is the quantity a selection rule actually exploits: how well the signal orders the draws.

Resource signals are negated before correlating, because the rules that use them prefer fewer
qubits and shorter chains, so a positive correlation means the rule is pointing the right way.
"""
import argparse, json, math, pathlib, statistics


def spearman(a, b):
    """Rank correlation with averaged ranks inside ties; ties are the common case for qubits."""
    n = len(a)
    if n < 3:
        return None

    def rank(xs):
        order = sorted(range(n), key=lambda i: xs[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    if den <= 0:                           # a signal with no variation orders nothing
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / den


SIGNALS = {
    "fewest qubits": lambda r: [-q for q in r["qubits"]],
    "shortest chain": lambda r: [-c for c in r["chain"]],
    "256-read measurement": lambda r: r["select"],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="selection-rule logs, one a host")
    a = ap.parse_args()
    print("  %-12s %-22s %5s %9s %22s %9s"
          % ("host", "signal", "n", "mean rho", "95 percent interval", "spread"), flush=True)
    for path in a.logs:
        p = pathlib.Path(path)
        rows = [json.loads(l) for l in p.read_text().splitlines()
                if l.startswith("{") and '"assess"' in l]
        host = p.stem
        for name, get in SIGNALS.items():
            rs = [spearman(get(r), r["assess"]) for r in rows]
            rs = [x for x in rs if x is not None]
            if not rs:
                print("  %-12s %-22s %5d %9s %22s" % (host, name, 0, "-", "no variation"), flush=True)
                continue
            n = len(rs)
            m = sum(rs) / n
            h = 1.96 * (statistics.stdev(rs) if n > 1 else 0.0) / math.sqrt(n)
            spread = (sum(max(get(r)) - min(get(r)) for r in rows) / len(rows))
            print("  %-12s %-22s %5d %9.3f        [%+.3f, %+.3f] %9.2f"
                  % (host, name, n, m, m - h, m + h, spread), flush=True)
            print(json.dumps({"host": host, "signal": name, "instances": n, "mean_rho": m,
                              "ci": [m - h, m + h], "mean_spread": spread,
                              "instances_with_variation": n, "instances_total": len(rows)}),
                  flush=True)
    print("SIGNAL RANKING DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
