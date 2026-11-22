"""Judge the S2 gate for two measurement campaigns (issue #236).

Usage: python judge_gate.py <campaign1.csv> <campaign2.csv>

The gate in force since 2026-11-22: S2 E2 median <= 600 ms, IQR/median <= 30%,
difference between the two campaign medians <= 10%. S4 stays at 300 ms and
150 ms for the increment. The 400 ms gate is history and is reported too.
"""
import csv
import math
import statistics as st
import sys


def load(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    out = {}
    for cond in ("E0", "E1", "E2", "E3"):
        out[cond] = {k: [float(r[k]) for r in rows if r["cond"] == cond]
                     for k in ("S1", "S2", "S3", "S4")}
    return out


def med(v):
    return st.median(v)


def p95(v):
    s = sorted(v)
    return s[math.ceil(0.95 * len(s)) - 1]


def iqr_over_med(v):
    s = sorted(v)
    n = len(s)
    return (st.median(s[(n + 1) // 2:]) - st.median(s[:n // 2])) / st.median(s)


def drift(v):
    return med(v[10:]) - med(v[:10]) if len(v) >= 20 else 0.0


def report(name, data):
    print("== %s" % name)
    for cond in ("E0", "E1", "E2", "E3"):
        v = data[cond]["S2"]
        print("  S2 %s median=%.1f p95=%.1f IQR/median=%.1f%% max=%.1f" % (
            cond, med(v), p95(v), 100 * iqr_over_med(v), max(v)))
    print("  S4 E2 median=%.1f p95=%.1f" % (med(data["E2"]["S4"]), p95(data["E2"]["S4"])))


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    a = load(sys.argv[1])
    b = load(sys.argv[2])
    report("campaign 1", a)
    report("campaign 2", b)

    e2a, e2b = med(a["E2"]["S2"]), med(b["E2"]["S2"])
    checks = [
        ("S2 E2 median <= 600 (campaign 1)", e2a <= 600.0, "%.1f ms" % e2a),
        ("S2 E2 median <= 600 (campaign 2)", e2b <= 600.0, "%.1f ms" % e2b),
        ("S2 IQR/median <= 30% (campaign 1)", iqr_over_med(a["E2"]["S2"]) <= 0.30,
         "%.1f%%" % (100 * iqr_over_med(a["E2"]["S2"]))),
        ("S2 IQR/median <= 30% (campaign 2)", iqr_over_med(b["E2"]["S2"]) <= 0.30,
         "%.1f%%" % (100 * iqr_over_med(b["E2"]["S2"]))),
        ("campaign medians differ <= 10%", abs(e2a - e2b) <= 0.10 * max(e2a, e2b),
         "%.1f ms (%.1f%%)" % (abs(e2a - e2b), 100 * abs(e2a - e2b) / max(e2a, e2b))),
        ("S4 E2 median <= 300", med(b["E2"]["S4"]) <= 300.0, "%.1f ms" % med(b["E2"]["S4"])),
        ("S4 E2-E0 <= 150", med(b["E2"]["S4"]) - med(b["E0"]["S4"]) <= 150.0,
         "%.1f ms" % (med(b["E2"]["S4"]) - med(b["E0"]["S4"]))),
    ]
    # the S2 increment keeps the C-24 drift rule
    inc = med(b["E2"]["S2"]) - med(b["E0"]["S2"])
    dr = max(abs(drift(b["E2"]["S2"])), abs(drift(b["E0"]["S2"])))
    if dr > 0.2 * abs(inc):
        checks.append(("S2 E2-E0 verdict", None, "%.1f ms, undecidable (drift %.1f ms)" % (inc, dr)))
    else:
        checks.append(("S2 E2-E0 <= 300", inc <= 300.0, "%.1f ms" % inc))

    print("\n== gate")
    ok = True
    for label, passed, value in checks:
        state = "PASS" if passed else ("UNDECIDED" if passed is None else "FAIL")
        if passed is False:
            ok = False
        print("  %-42s %-9s %s" % (label, state, value))
    print("\n  400 ms gate (history): campaign 1 %.1f ms -> %s" % (
        e2a, "FAIL" if e2a > 400.0 else "PASS"))
    print("  formal pass needs the two campaigns to be at least two days apart")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
