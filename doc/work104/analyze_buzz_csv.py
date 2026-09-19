"""WORK104 helper: summarize BassBuzzMeasurement CSV + cross-check WORK103 predictions."""
import csv, sys, math

path = sys.argv[1] if len(sys.argv) > 1 else "buzz_results.csv"
rows = list(csv.DictReader(open(path)))
print(f"rows={len(rows)} file={path}")
print(f"{'cfg':4} {'sig':8} {'inPk':>7} {'outPk':>7} {'crest':>6} {'flat':>6} {'lim':>6} {'thd':>8} {'ultra':>8}")
for r in rows:
    print(f"{r['config']:4} {r['signal']:8} {float(r['inPeak']):7.4f} {float(r['outPeak']):7.4f} "
          f"{float(r['crestDb']):6.1f} {int(r['flatTop']):6d} {int(r['limitZone']):6d} "
          f"{float(r['thdDb']):8.1f} {float(r['ultraRatioDb']):8.1f}")

def get(cfg, sig):
    for r in rows:
        if r["config"] == cfg and r["signal"] == sig:
            return r
    return None

print("---- verdict cross-check ----")
a = get("C3", "sine50")
a4 = get("C4", "sine50")
if a:
    engaged = int(a["flatTop"]) > 0 or float(a["thdDb"]) > -40.0
    print(f"A: C3 sine50 flat={a['flatTop']} thd={a['thdDb']} -> {'ENGAGED' if engaged else 'CLEAR'}")
    if a4:
        cleared = int(a4["flatTop"]) == 0 and float(a4["thdDb"]) < float(a["thdDb"]) - 10.0
        print(f"A: C4 clears -> {'YES' if cleared else 'NO'} (flat={a4['flatTop']} thd={a4['thdDb']})")
c0 = get("C0", "sine50"); c1 = get("C1", "sine50")
if c0 and c1:
    print(f"C: ultra C0={float(c0['ultraRatioDb']):.1f}dB C1={float(c1['ultraRatioDb']):.1f}dB "
          f"delta={float(c1['ultraRatioDb']) - float(c0['ultraRatioDb']):+.1f}dB")
b0 = get("C0", "sine50")
if b0:
    print(f"rig: C0 sine50 thd={b0['thdDb']}dB outPeak={b0['inPeak']}->{b0['outPeak']} "
          f"(expect thd<-90, peak~=in for transparency)")
