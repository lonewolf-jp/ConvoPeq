"""#236 campaign 2: the same E0-E3 20-round balanced Latin square, on the same binary.

The C-24 plan requires the second campaign to be at least two days after the
first one, so this script refuses to run earlier than that and refuses to reuse
the campaign-1 CSV.

Usage:
    python campaign2.py [rounds]

Writes campaign236_c2.csv next to this script. Then judge with:
    python judge_gate.py campaign236.csv campaign236_c2.csv
"""
import datetime
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST_DATE = os.path.join(HERE, "campaign1_date.txt")
CAMPAIGN1 = os.path.join(HERE, "campaign236.csv")
OUT = os.path.join(HERE, "campaign236_c2.csv")

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


def first_campaign_date():
    with open(FIRST_DATE, encoding="utf-8") as fh:
        text = fh.read().strip()
    return datetime.date.fromisoformat(text)


def main():
    if not os.path.exists(CAMPAIGN_PY):
        raise SystemExit("campaign.py is missing: %s" % CAMPAIGN_PY)
    if not os.path.exists(EXE):
        raise SystemExit("measurement binary is missing: %s" % EXE)

    first = first_campaign_date()
    today = datetime.date.today()
    gap = (today - first).days
    if gap < 2:
        raise SystemExit(
            "campaign 2 must be at least 2 days after campaign 1 (%s): today is %s "
            "(%d day(s)). Nothing was measured." % (first, today, gap))

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
    print("campaign 2: gap=%d days, exe=%s, out=%s" % (gap, EXE, OUT))
    campaign.main()


if __name__ == "__main__":
    main()
