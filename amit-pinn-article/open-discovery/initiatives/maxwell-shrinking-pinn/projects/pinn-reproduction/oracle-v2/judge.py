"""Apply runs/OPEN-ITEMS-PREREGISTRATION.md item-B criteria to table3_oracle.json (no recomputation)."""
import json, sys
d = json.load(open(sys.argv[1])); Ls = [str(float(x)) for x in d["domains"]]
rel = lambda a, b: abs(a / b - 1)
print("| S | lower a1 (L=24) | shoot/cheb/asym agree | dL20->24 | converged | paper lower | upper f_inf | alpha* | upper lowest root (shoot) L=12/16/20/24 | dL20->24 | vs alpha* | swapped L=12 upper (err vs paper) | min energy eig upper / lower (L=24) |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in d["rows"]:
    lo, up = r["lower"], r["upper"]; L1, L2 = Ls[-2], Ls[-1]
    vals = [lo["by_L"][L2][m][0] for m in ("shoot", "cheb", "asym")]
    agree = max(rel(v, vals[0]) for v in vals) <= 0.005
    dL = rel(lo["by_L"][L2]["shoot"][0], lo["by_L"][L1]["shoot"][0])
    conv = agree and dL < 0.005
    us = [up["by_L"][L]["shoot"][0] for L in Ls[1:]]
    du = rel(us[-1], us[-2]); a_s = up["alpha_star"]
    sw = up["by_L"]["12.0"]["swapped_bundled"][0]
    print(f"| {r['S']} | {vals[0]:.5f} | {'yes' if agree else 'NO'} ({max(rel(v, vals[0]) for v in vals):.1e}) | {dL:.1e} | {'yes' if conv else 'NO'} | {r['paper']['lower']} "
          f"| {up['f_inf']:.4f} | {a_s:.4f} | {' / '.join(f'{u:.4f}' for u in us)} | {du:.2%} | {'below' if us[-1] < a_s else 'above'} ({us[-1]/a_s-1:+.1%}) "
          f"| {sw:.5f} ({rel(sw, r['paper']['upper']):.2%}) | {up['by_L'][L2]['energy_cheb'][0]:.4f} / {lo['by_L'][L2]['energy_cheb'][0]:.4f} |")
