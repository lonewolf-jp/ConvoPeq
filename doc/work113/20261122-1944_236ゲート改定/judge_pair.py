"""Judge the paired E0/E2 verification for issue #236.

The design and these rules were fixed before the run (campaign_pair.py and
report.md 7.3):

  PASS      upper bound of the 95% bootstrap CI of median d_i <= 300 ms
  UNDECIDED the CI straddles 300 ms
  FAIL      lower bound of the 95% CI >= 300 ms

  d_i = S2(E2) - S2(E0) inside pair i.

Order effect: the medians of the two order groups are compared. When they
differ by more than 20 ms the rule is applied per group and PASS needs both.

Drift: a regression of d_i on the pair index over the whole span and the
first-half/second-half median difference are printed as diagnostics. They do
not override the CI: the bootstrap resamples the observed pair differences, so
pair-to-pair variation (the thing the C-24 20% rule guards against) is already
inside the interval.

Exit codes: 0 PASS, 1 FAIL, 2 UNDECIDED, 3 unusable data.
"""
import csv
import math
import random
import statistics as st
import sys

PAIRS = 20
LIMIT_MS = 300.0
ORDER_LIMIT_MS = 20.0
BOOTSTRAP = 10000
SEED = 20261123


def unusable(message):
    print("ERROR: %s" % message, file=sys.stderr)
    raise SystemExit(3)


def load(path):
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
        columns = rows[0].keys() if rows else []
    for need in ("pair", "order", "cond", "S2"):
        if need not in columns:
            unusable("%s: missing column %s" % (path, need))
    pairs = {}
    for row in rows:
        try:
            pair = int(row["pair"])
            s2 = float(row["S2"])
        except (TypeError, ValueError):
            unusable("%s: pair %r has an unusable S2 %r" % (path, row.get("pair"), row.get("S2")))
        if not math.isfinite(s2) or s2 <= 0:
            unusable("%s: pair %d has an invalid S2 %r" % (path, pair, row["S2"]))
        pairs.setdefault(pair, {})[row["cond"]] = (s2, row["order"])
    if len(pairs) != PAIRS:
        unusable("%s: %d pairs, expected %d" % (path, len(pairs), PAIRS))
    diffs = []
    for pair in sorted(pairs):
        legs = pairs[pair]
        if set(legs) != {"E0", "E2"}:
            unusable("%s: pair %d does not hold exactly one E0 and one E2" % (path, pair))
        d = legs["E2"][0] - legs["E0"][0]
        diffs.append((pair, legs["E0"][1], d))
    return diffs


def bootstrap_ci(values, rng):
    medians = []
    n = len(values)
    for _ in range(BOOTSTRAP):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        medians.append(st.median(sample))
    medians.sort()
    return medians[int(0.025 * BOOTSTRAP)], medians[int(0.975 * BOOTSTRAP) - 1]


def regression_drift(values):
    n = len(values)
    xs = list(range(n))
    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values))
    return (sxy / sxx if sxx else 0.0) * (n - 1)


def verdict(values, rng, label):
    lo, hi = bootstrap_ci(values, rng)
    med = st.median(values)
    # the gate is "300 ms or less", so the boundary belongs to PASS
    if hi <= LIMIT_MS:
        state = "PASS"
    elif lo > LIMIT_MS:
        state = "FAIL"
    else:
        state = "UNDECIDED"
    print("  %-28s median=%6.2f ms  CI95=[%6.2f, %6.2f]  %s"
          % (label, med, lo, hi, state))
    return state


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(3)
    diffs = load(sys.argv[1])
    rng = random.Random(SEED)

    print("== pairs")
    for pair, order, d in diffs:
        print("  pair %02d %-7s d=%+7.2f ms" % (pair, order, d))

    values = [d for _, _, d in diffs]
    print("\n== drift (diagnostics, not a verdict)")
    print("  regression over the span: %+.2f ms" % regression_drift(values))
    print("  second half - first half: %+.2f ms"
          % (st.median(values[10:]) - st.median(values[:10])))

    groups = {}
    for _, order, d in diffs:
        groups.setdefault(order, []).append(d)
    print("\n== order effect")
    group_medians = {}
    for order in sorted(groups):
        group_medians[order] = st.median(groups[order])
        print("  %-7s n=%d median=%+7.2f ms" % (order, len(groups[order]), group_medians[order]))
    order_gap = abs(group_medians["E0->E2"] - group_medians["E2->E0"])

    print("\n== gate (S2 increment, paired)")
    if order_gap > ORDER_LIMIT_MS:
        print("  order medians differ by %.1f ms (> %.0f): judged per group"
              % (order_gap, ORDER_LIMIT_MS))
        states = [verdict(groups[o], rng, "order " + o) for o in sorted(groups)]
        if "FAIL" in states:
            state = "FAIL"
        elif all(s == "PASS" for s in states):
            state = "PASS"
        else:
            state = "UNDECIDED"
    else:
        state = verdict(values, rng, "all pairs")

    print("\n  judge_pair verdict: %s (limit %.0f ms)" % (state, LIMIT_MS))
    if state == "PASS":
        raise SystemExit(0)
    if state == "FAIL":
        raise SystemExit(1)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
