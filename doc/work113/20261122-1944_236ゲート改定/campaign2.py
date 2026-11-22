"""#236 campaign 2: the same E0-E3 20-round balanced Latin square, on the same binary.

Campaign 1 finished on 2026-11-22 19:44. The original C-24 plan required the
second campaign at least two days after the first one. On 2026-11-23 the owner
changed the minimum interval to four hours, to shorten the turnaround. Four
hours does not give the temporal independence of two days: campaign 1 and the
same-day reference run (campaign 1b) are close in time, so campaign 2 is a
short-term repeat measurement, not a long-term reproducibility check. See
doc/work113/20261122-1944_236ゲート改定/report.md.

Usage:
    python campaign2.py [rounds]   run the campaign, writes campaign236_c2.csv
    python campaign2.py --check    evaluate the interval guard and exit

The guard is measured from campaign1_end.txt (campaign 1 finish time), never
from a calendar date, and it refuses to start below four hours. The campaign-1
CSV is never overwritten and an existing campaign-2 CSV stops the run.

Then judge with:
    python judge_gate.py campaign236.csv campaign236_c2.csv
"""
import datetime
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST_END = os.path.join(HERE, "campaign1_end.txt")
CAMPAIGN1 = os.path.join(HERE, "campaign236.csv")
OUT = os.path.join(HERE, "campaign236_c2.csv")

# Minimum gap between the end of campaign 1 and the start of campaign 2.
# 2026-11-23: changed from two days (C-24 plan) to four hours by owner decision.
MIN_GAP = datetime.timedelta(hours=4)

# Campaign 2 must measure the same binary as campaign 1. The instrumented build
# of the merged change is kept outside the repository; override with
# CONVOPEQ_C2_EXE when the binary is elsewhere.
EXE = os.environ.get(
    "CONVOPEQ_C2_EXE",
    r"C:\Users\user\AppData\Local\Temp\opencode\m236\exe_post.exe")

# campaign.py is shipped next to this file and holds the rounding, condition
# restore, trace parsing and Latin-square order. Import it instead of copying,
# so both campaigns cannot drift apart.
CAMPAIGN_PY = os.path.join(HERE, "campaign.py")


def campaign1_end():
    with open(FIRST_END, encoding="utf-8") as fh:
        text = fh.read().strip()
    try:
        return datetime.datetime.strptime(text, "%Y-%m-%d %H:%M")
    except ValueError:
        raise SystemExit("campaign1_end.txt must hold 'YYYY-MM-DD HH:MM', got %r" % text)


def main():
    check_only = "--check" in sys.argv[1:]

    if not os.path.exists(CAMPAIGN_PY):
        raise SystemExit("campaign.py is missing: %s" % CAMPAIGN_PY)
    if not os.path.exists(EXE):
        raise SystemExit("measurement binary is missing: %s" % EXE)

    end = campaign1_end()
    now = datetime.datetime.now()
    gap = now - end
    print("campaign 1 finished %s; now %s; gap=%s (minimum %s)"
          % (end.strftime("%Y-%m-%d %H:%M"), now.strftime("%Y-%m-%d %H:%M"),
             gap, MIN_GAP))
    if gap < MIN_GAP:
        message = ("campaign 2 must start at least 4 h after campaign 1 finished "
                   "(%s); the gap is %s. Nothing was measured." % (end, gap))
        if check_only:
            print("REFUSED: %s" % message)
            raise SystemExit(1)
        raise SystemExit(message)

    if check_only:
        print("CHECK OK: the 4 h guard allows the campaign")
        return

    spec = importlib.util.spec_from_file_location("campaign236", CAMPAIGN_PY)
    campaign = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(campaign)

    if os.path.abspath(campaign.OUT) == os.path.abspath(OUT):
        raise SystemExit("refusing to overwrite the campaign-1 CSV")
    if os.path.exists(OUT):
        # keep the independent record: never append to or overwrite a campaign
        raise SystemExit("campaign236_c2.csv already exists; move it away first")

    campaign.EXE = EXE
    campaign.OUT = OUT
    print("campaign 2: exe=%s, out=%s" % (EXE, OUT))
    campaign.main()


if __name__ == "__main__":
    main()
