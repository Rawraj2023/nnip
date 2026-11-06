"""Item A decision rule (runs/OPEN-ITEMS-PREREGISTRATION.md) applied to confirm_fresh.py outputs.
Usage: python analyze.py baseline.json c005.json [session2_c005.json]"""
import json, sys, statistics as st
def pts(path):
    d = json.load(open(path)); return [(r["params"]["S"], r["params"]["K"], r["params"]["beta"], run) for r in d["rows"] for run in r["runs"]]
B, C = pts(sys.argv[1]), pts(sys.argv[2])
def verdict(mb, m5, xb, x5):
    if m5 > mb + 0.20 or x5 > xb + 0.30: return "005 WORSE"
    if m5 < mb - 0.20: return "005 better"
    return "equivalent (no gross difference)"
print("| metric | baseline mean / max (%) | 005 mean / max (%) | verdict |\n|---|---|---|---|")
for br, q in (("lower", "Cf"), ("upper", "Cf"), ("upper", "Nu"), ("lower", "Nu")):
    eb = [100 * p[3][br][q + "_err_oracle"] for p in B]; e5 = [100 * p[3][br][q + "_err_oracle"] for p in C]
    tag = " (PRIMARY)" if (br, q) == ("lower", "Cf") else ""
    print(f"| {br} {q}{tag} | {st.mean(eb):.3f} / {max(eb):.3f} | {st.mean(e5):.3f} / {max(e5):.3f} | {verdict(st.mean(eb), st.mean(e5), max(eb), max(e5))} |")
print("\nPer point, lower-branch Cf error vs oracle (%):\n\n| S | K | beta | seed | baseline | 005 (this GPU) |" + (" 005 (session 2, 1080 Ti) |" if len(sys.argv) > 3 else "") + "\n|---|---|---|---|---|---|" + ("---|" if len(sys.argv) > 3 else ""))
S2 = pts(sys.argv[3]) if len(sys.argv) > 3 else None
for i, (b, c) in enumerate(zip(B, C)):
    extra = f" {100 * S2[i][3]['lower']['Cf_err_oracle']:.3f} |" if S2 else ""
    print(f"| {b[0]} | {b[1]} | {b[2]} | {b[3]['seed']} | {100 * b[3]['lower']['Cf_err_oracle']:.3f} | {100 * c[3]['lower']['Cf_err_oracle']:.3f} |{extra}")
tb, tc = sum(p[3]["seconds"] for p in B), sum(p[3]["seconds"] for p in C)
print(f"\nWall-clock over the 9 fresh points (informative, this GPU): baseline {tb:.0f} s, 005 {tc:.0f} s, ratio {tb / tc:.2f}x")
