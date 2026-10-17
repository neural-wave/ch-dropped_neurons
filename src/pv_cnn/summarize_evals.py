"""Table of every artifacts/eval_*.json: score, F1, D, P and the misclassified ids.   python summarize_evals.py [pattern]"""
import json, sys
from config import ARTIFACT_DIR

pat = sys.argv[1] if len(sys.argv) > 1 else ""
print(f"{'run':30s} {'score':>6s} {'F1':>6s} {'D':>5s} {'P':>5s} {'n':>3s}  false PV / missed PV")
for p in sorted(ARTIFACT_DIR.glob("eval_*.json")):
    if pat not in p.name:
        continue
    r = json.load(open(p)); pc = r.get("per_customer", [])
    fp = [x["customer_id"] for x in pc if not x["pv_installed_true"] and x["pv_installed_pred"]]
    fn = [x["customer_id"] for x in pc if x["pv_installed_true"] and not x["pv_installed_pred"]]
    print(f"{p.stem[5:]:30s} {r['score']:6.1f} {r['F1']:6.3f} {r['D']:5.2f} {r['P']:5.2f} {len(pc):3d}  FP {','.join(fp) or '-'} / FN {','.join(fn) or '-'}")
