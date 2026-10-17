"""Table of every artifacts/eval_*.json: score, F1, D, P and the misclassified customer ids.
    python summarize_evals.py [pattern]      e.g.  python summarize_evals.py loocv"""
import json, sys
from config import ARTIFACT_DIR

pat = sys.argv[1] if len(sys.argv) > 1 else ""
rows = []
for p in sorted(ARTIFACT_DIR.glob("eval_*.json")):
    if pat not in p.name:
        continue
    r = json.load(open(p)); pc = r.get("per_customer", [])
    fp = [x["customer_id"] for x in pc if not x["pv_installed_true"] and x["pv_installed_pred"]]
    fn = [x["customer_id"] for x in pc if x["pv_installed_true"] and not x["pv_installed_pred"]]
    rows.append((p.stem[5:], r.get("score", float("nan")), r.get("F1", float("nan")), r.get("D", float("nan")), r.get("P", float("nan")), len(pc), fp, fn))
print(f"{'run':38s} {'score':>6s} {'F1':>6s} {'D':>5s} {'P':>5s} {'n':>3s}  false PV / missed PV")
for name, sc, f1, D, P, n, fp, fn in rows:
    print(f"{name:38s} {sc:6.1f} {f1:6.3f} {D:5.2f} {P:5.2f} {n:3d}  FP {','.join(fp) or '-'} / FN {','.join(fn) or '-'}")
