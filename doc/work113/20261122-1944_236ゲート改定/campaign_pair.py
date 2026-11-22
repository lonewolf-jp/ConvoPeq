"""#236 paired E0/E2 verification. The design was fixed before the run.

Why: the two 20-round campaigns leave the S2 increment (E2 - E0) undecided,
because the increment is about 10 ms while the drift of a 20-run sequence is
20 ms or more. Pairing E0 and E2 back to back removes most of that drift from
the difference, so the increment can be judged on its own.

Fixed design (do not change after seeing data):
- 20 pairs. Odd pairs run E0 then E2; even pairs run E2 then E0 (10 each), so
  both orders are spread across the whole period.
- The two legs of a pair run back to back; 2 s between pairs. The gap between
  the end of one leg and the start of the next is recorded.
- Every run restores that condition's files and records size and SHA-256.
- Same binary, fixtures, CLI options and trace parsing as campaigns 1 and 2.
- d_i = S2(E2) - S2(E0) per pair is the primary measurement.

Judgment (judge_pair.py; fixed before the run):
- PASS      the upper bound of the 95% bootstrap CI of median d_i is < 300 ms.
- UNDECIDED the CI straddles 300 ms.
- FAIL      the lower bound of the 95% CI is >= 300 ms.
- Order effect: medians per order group and the CI of their difference. If the
  medians differ by more than 20 ms, the rule is applied to each group and PASS
  needs both groups to pass.
- Drift: regression of d_i on the pair index over the whole span, plus the
  first-half/second-half median difference. Reported as diagnostics; the
  bootstrap CI already carries the pair-to-pair variation that the C-24 20%
  rule was protecting against.

Usage:
    python campaign_pair.py [pairs]      run (default 20 pairs)
    python judge_pair.py campaign236_pairs.csv
"""
import csv
import importlib.util
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profile_guard

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "campaign236_pairs.csv")
PAIRS = 20
GAP_BETWEEN_PAIRS_S = 2.0

# Same instrumented binary as campaigns 1 and 2. campaign.py's default points at
# the ordinary MSVC build, which has no timing marks.
EXE = os.environ.get(
    "CONVOPEQ_C2_EXE",
    r"C:\Users\user\AppData\Local\Temp\opencode\m236\exe_post.exe")


def load_campaign():
    spec = importlib.util.spec_from_file_location(
        "campaign236", os.path.join(HERE, "campaign.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    pairs = int(sys.argv[1]) if len(sys.argv) > 1 else PAIRS
    if pairs % 2 != 0:
        raise SystemExit("pairs must be even so both orders get the same count")
    if os.path.exists(OUT):
        raise SystemExit("%s already exists; move it away first" % OUT)

    campaign = load_campaign()
    if not os.path.exists(EXE):
        raise SystemExit("measurement binary is missing: %s" % EXE)
    campaign.EXE = EXE

    columns = ["pair", "order", "leg", "cond", "start_epoch", "end_epoch",
               "gap_ms", "S1", "S2", "S3", "S4", "S2_cpu", "init_cpu",
               "nsl", "nsl_sha", "ls", "ls_sha", "ds_sha"]

    guard = profile_guard.Guard()
    guard.save()
    previous_end = None
    try:
        with open(OUT, "w", encoding="utf-8", newline="\n") as out:
            out.write(",".join(columns) + "\n")
            for pair in range(1, pairs + 1):
                order = ["E0", "E2"] if pair % 2 == 1 else ["E2", "E0"]
                for leg, cond in enumerate(order):
                    gap_ms = -1.0
                    if previous_end is not None:
                        gap_ms = (time.time() - previous_end) * 1000.0
                    start = time.time()
                    row = campaign.run_once(cond, pair - 1, leg)
                    end = time.time()
                    previous_end = end
                    values = [str(pair), "->".join(order), str(leg + 1), cond,
                              "%.3f" % start, "%.3f" % end, "%.0f" % gap_ms]
                    values += ["%.1f" % row[k] for k in
                               ("S1", "S2", "S3", "S4", "S2_cpu", "init_cpu")]
                    values += [str(row["nsl"]), row["nsl_sha"],
                               str(row["ls"]), row["ls_sha"], row["ds_sha"]]
                    out.write(",".join(values) + "\n")
                    out.flush()
                    print("pair %02d leg %d %-2s S2=%7.1f gap=%6.0f ms"
                          % (pair, leg + 1, cond, row["S2"], gap_ms), flush=True)
                time.sleep(GAP_BETWEEN_PAIRS_S)
    finally:
        guard.restore()

    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
