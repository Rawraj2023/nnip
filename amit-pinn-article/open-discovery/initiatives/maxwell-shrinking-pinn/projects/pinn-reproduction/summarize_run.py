"""Print a one-screen summary of an evaluator full-mode JSON (reads output only)."""
import json, sys
TRUE = {"upper": (1.93525916, 3.35132041), "lower": (0.75370240, 3.13540252)}  # oracle, eta_inf=40
for path in sys.argv[1:]:
    d = json.load(open(path)); m = d["metrics"]; t = m["table1_2"]
    print(f"{path}: valid={d['valid']} improved={d['improved']} objective={d['objective']['value']:.1f}s "
          f"(baseline {d['objective']['baseline_value']:.1f}s) failures={d['failures']}")
    print(f"  table1/2 {t['elapsed_seconds']:.1f}s  " + "  ".join(f"{k}={v:.4%}" for k, v in t["errors"].items()))
    for br in ("upper", "lower"):
        print(f"  {br}: Cf={t[br]['Cf']:.6f} (vs eta_inf->inf oracle {abs(t[br]['Cf']/TRUE[br][0]-1):.3%})"
              f"  Nu={t[br]['Nu']:.6f} ({abs(t[br]['Nu']/TRUE[br][1]-1):.3%})")
    if "fold" in m:
        print(f"  fold {m['fold']['elapsed_seconds']:.1f}s  S_c={m['fold']['S_c']:.5f} err={m['fold']['error']:.3%}")
    if "stability" in m:
        s = m["stability"]
        print(f"  stability {s['elapsed_seconds']:.1f}s  a_up={s['alpha1_upper']:.6f} a_lo={s['alpha1_lower']:.6f} "
              + " ".join(f"{k}={v:.3%}" for k, v in s["errors"].items()))
