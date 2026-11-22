"""Judge the S2 gate for two measurement campaigns (issue #236).

Usage: python judge_gate.py <campaign1.csv> <campaign2.csv>

The gate in force since 2026-11-22: S2 E2 median <= 600 ms, IQR/median <= 30%,
difference between the two campaign medians <= 10%. S4 stays at 300 ms and
150 ms for the increment. The 400 ms gate is history and is reported too.

Exit codes: 0 all conditions pass, 1 a measured condition fails, 2 a condition
is undecided (drift makes the S2 increment unusable), 3 the input data is not
usable (missing marks, non-finite or negative timings, wrong row count).

The S2 increment uses the C-24 drift rule with both estimators the plan names:
the first-half/second-half median difference and a regression of the value on
its measured position. The larger magnitude is compared with 20% of the
increment.
"""
import csv
import math
import statistics as st
import sys

CONDITIONS = ("E0", "E1", "E2", "E3")
FIELDS = ("S1", "S2", "S3", "S4")
ROWS_PER_CONDITION = 20


def unusable(message):
    # 3 = the CSV cannot support a verdict (distinct from a measured failure)
    print("ERROR: %s" % message, file=sys.stderr)
    raise SystemExit(3)


def load(path):
    with open(path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        columns = reader.fieldnames or []

    missing = [c for c in ("cond",) + FIELDS if c not in columns]
    if missing:
        unusable("%s: missing column(s) %s" % (path, ", ".join(missing)))

    data = {}
    for cond in CONDITIONS:
        selected = [r for r in rows if r["cond"] == cond]
        if len(selected) != ROWS_PER_CONDITION:
            unusable("%s: %s has %d rows, expected %d"
                     % (path, cond, len(selected), ROWS_PER_CONDITION))
        data[cond] = {}
        for field in FIELDS:
            values = []
            for r in selected:
                try:
                    v = float(r[field])
                except (TypeError, ValueError):
                    unusable("%s: %s %s is not a number" % (path, cond, field))
                # campaign.py writes -1.0 when a timing mark is missing
                if not math.isfinite(v) or v <= 0.0:
                    unusable("%s: %s %s has a missing or invalid timing (%r)"
                             % (path, cond, field, r[field]))
                values.append(v)
            data[cond][field] = values
    return data


def med(v):
    return st.median(v)


def p95(v):
    s = sorted(v)
    return s[math.ceil(0.95 * len(s)) - 1]


def iqr_over_med(v):
    s = sorted(v)
    n = len(s)
    return (st.median(s[(n + 1) // 2:]) - st.median(s[:n // 2])) / st.median(s)


def half_drift(v):
    """Drift estimator 1 (C-24): median of the second half minus the first half."""
    return med(v[10:]) - med(v[:10])


def regression_drift(v):
    """Drift estimator 2 (C-24): the slope of the value against its measured
    position, expressed over the whole measured span."""
    n = len(v)
    xs = list(range(n))
    mean_x = (n - 1) / 2.0
    mean_y = sum(v) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, v))
    slope = sxy / sxx if sxx else 0.0
    return slope * (n - 1)


def drift(v):
    """C-24 evaluates drift with both estimators; the larger magnitude is used,
    so a slow trend that hides inside one half cannot pass unnoticed."""
    return max(abs(half_drift(v)), abs(regression_drift(v)))


def report(name, data):
    print("== %s" % name)
    for cond in CONDITIONS:
        v = data[cond]["S2"]
        print("  S2 %s median=%.1f p95=%.1f IQR/median=%.1f%% max=%.1f" % (
            cond, med(v), p95(v), 100 * iqr_over_med(v), max(v)))
    print("  S4 E2 median=%.1f p95=%.1f" % (med(data["E2"]["S4"]), p95(data["E2"]["S4"])))


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(3)
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

    # the S2 increment keeps the C-24 drift rule: both estimators, larger wins
    inc = med(b["E2"]["S2"]) - med(b["E0"]["S2"])
    dr = max(drift(b["E2"]["S2"]), drift(b["E0"]["S2"]))
    print("  S2 drift (campaign 2): E2 half=%.1f reg=%.1f | E0 half=%.1f reg=%.1f"
          % (half_drift(b["E2"]["S2"]), regression_drift(b["E2"]["S2"]),
             half_drift(b["E0"]["S2"]), regression_drift(b["E0"]["S2"])))
    if dr > 0.2 * abs(inc):
        checks.append(("S2 E2-E0 verdict", None,
                       "%.1f ms, undecidable (drift %.1f ms > 20%% of %.1f ms)"
                       % (inc, dr, abs(inc))))
    else:
        checks.append(("S2 E2-E0 <= 300", inc <= 300.0, "%.1f ms" % inc))

    print("\n== gate")
    failed = False
    undecided = False
    for label, passed, value in checks:
        state = "PASS" if passed else ("UNDECIDED" if passed is None else "FAIL")
        if passed is False:
            failed = True
        elif passed is None:
            undecided = True
        print("  %-42s %-9s %s" % (label, state, value))

    print("\n  400 ms gate (history): campaign 1 %.1f ms -> %s" % (
        e2a, "FAIL" if e2a > 400.0 else "PASS"))
    print("  formal pass needs the two campaigns to be at least two days apart")

    if failed:
        raise SystemExit(1)
    if undecided:
        # an undecided condition is not a pass: the caller must not read 0 here
        raise SystemExit(2)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
