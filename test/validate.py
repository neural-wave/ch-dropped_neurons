"""validate.py test_csv out_dir labels_csv ext_labels_csv  -> checks PDF 4.1 format rules and computes 4.2/4.3 scores."""
import csv, re, sys
from datetime import date

test_csv, out_dir, labels_csv, ext_csv = sys.argv[1:5]
fails = []
def check(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond: fails.append(msg)

def read_semicolon(path):
    raw = open(path, "rb").read()
    raw.decode("utf-8")  # raises if not UTF-8
    bom = raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8")
    lines = text.split("\n")
    rows = [l.split(";") for l in lines if l != ""]
    return rows, bom, text

# customer ids of the test file, in order
ids = []
with open(test_csv, encoding="utf-8-sig", newline="") as f:
    r = csv.reader(f, delimiter=";"); next(r)
    seen = set()
    for row in r:
        if row[0] not in seen: seen.add(row[0]); ids.append(row[0])

DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
def valid_date(s):
    if not DATE.match(s): return False
    try: date.fromisoformat(s); return True
    except ValueError: return False
def valid_num(s):
    try: return float(s) > 0 and s.strip() == s and not s.startswith("+") and "," not in s
    except ValueError: return False

print("== predictions.csv")
rows, bom, text = read_semicolon(f"{out_dir}/predictions.csv")
check(not bom, "no UTF-8 BOM")
check("\r" not in text, "no CR line endings")
check("," not in text, "no commas anywhere (semicolon separator only, decimal point)")
check(rows[0] == ["customer_id", "pv_installed", "pv_installation_date", "pv_capacity_kwp"], f"exact header/order: {rows[0]}")
body = rows[1:]
check(all(len(r) == 4 for r in body), "every row has 4 fields")
check([r[0] for r in body] == ids, f"exactly one row per test customer, same ids ({len(body)} rows vs {len(ids)} ids), file order")
check(len({r[0] for r in body}) == len(body), "no duplicate customer_id")
check(all(r[1] in ("TRUE", "FALSE") for r in body), "pv_installed is TRUE/FALSE")
for r in body:
    cid, pv, d, k = r
    if pv == "TRUE":
        check(valid_date(d), f"{cid}: date YYYY-MM-DD valid ({d!r})")
        check(valid_num(k), f"{cid}: capacity numeric positive, decimal point ({k!r})")
    else:
        check(d == "" and k == "", f"{cid}: FALSE -> date and capacity empty ({d!r},{k!r})")
pred = {r[0]: r for r in body}

print("== pv_extensions_predictions.csv")
erows, ebom, etext = read_semicolon(f"{out_dir}/pv_extensions_predictions.csv")
check(not ebom, "no UTF-8 BOM")
check("," not in etext, "no commas")
check(erows[0] == ["customer_id", "extension_date", "pv_capacity_after_extension_kwp"], f"exact header/order: {erows[0]}")
ebody = erows[1:]
check(all(len(r) == 3 for r in ebody), "every row has 3 fields")
for r in ebody:
    check(r[0] in set(ids), f"{r[0]}: customer id in test set")
    check(valid_date(r[1]), f"{r[0]}: extension date valid ({r[1]!r})")
    check(valid_num(r[2]), f"{r[0]}: capacity after extension numeric positive ({r[2]!r})")
    check(r[0] in pred and pred[r[0]][1] == "TRUE", f"{r[0]}: extension only for a customer predicted TRUE")
print(f"  ({len(ebody)} extension rows)")

print("== scoring vs labels (PDF 4.2)")
lab = {}
with open(labels_csv, encoding="utf-8-sig") as f:
    r = csv.reader(f, delimiter=";"); next(r)
    for row in r: lab[row[0]] = row
tp = fp = fn = tn = 0; D = []; P = []
print(f"  {'id':6} {'true':20} {'pred':28} {'date_err':>8} {'kwp_ape':>8}")
for cid in ids:
    t = lab[cid]; p = pred[cid]
    tpv, ppv = t[1] == "TRUE", p[1] == "TRUE"
    de = ape = ""
    if tpv and ppv:
        tp += 1
        ed = abs((date.fromisoformat(p[2]) - date.fromisoformat(t[2])).days) if valid_date(p[2]) else None
        D.append(max(0, 1 - ed / 30) if ed is not None else 0)
        a = abs(float(p[3]) - float(t[3])) / float(t[3]) if valid_num(p[3]) else None
        P.append(max(0, 1 - a / 0.5) if a is not None else 0)
        de, ape = f"{ed}d", f"{a*100:.1f}%"
    elif tpv: fn += 1
    elif ppv: fp += 1
    else: tn += 1
    print(f"  {cid:6} {t[1]+' '+t[2]+' '+t[3]:20} {p[1]+' '+p[2]+' '+p[3]:28} {de:>8} {ape:>8}")
f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
Dm = sum(D) / len(D) if D else 0.0; Pm = sum(P) / len(P) if P else 0.0
print(f"  TP={tp} FP={fp} FN={fn} TN={tn}  F1={f1:.3f}  D={Dm:.3f}  P={Pm:.3f}  main score = {f1*(70+15*Dm+15*Pm):.2f}")

print("== extension scoring (PDF 4.3)")
ext_true = []
with open(ext_csv, encoding="utf-8-sig") as f:
    next(f)
    for line in f:
        parts = re.split(r"[;,]", line.strip())
        if len(parts) == 3 and parts[0] in set(ids): ext_true.append(parts)
preds = [r for r in ebody]
matched = []; used = set()
for t in ext_true:
    best = None
    for i, p in enumerate(preds):
        if i in used or p[0] != t[0] or not valid_date(p[1]): continue
        dd = abs((date.fromisoformat(p[1]) - date.fromisoformat(t[1])).days)
        if dd <= 30 and (best is None or dd < best[1]): best = (i, dd)
    if best: used.add(best[0]); matched.append((t, preds[best[0]], best[1]))
etp = len(matched); efp = len(preds) - etp; efn = len(ext_true) - etp
f1e = 2 * etp / (2 * etp + efp + efn) if (etp + efp + efn) else 0.0
De = sum(max(0, 1 - m[2] / 30) for m in matched) / etp if etp else 0
Pe = sum(max(0, 1 - abs(float(m[1][2]) - float(m[0][2])) / float(m[0][2]) / 0.5) if valid_num(m[1][2]) else 0 for m in matched) / etp if etp else 0
print(f"  true ext in sample: {ext_true}; predicted: {preds}")
print(f"  TP={etp} FP={efp} FN={efn}  F1_ext={f1e:.3f} D_ext={De:.3f} P_ext={Pe:.3f}  bonus = {f1e*(7+1.5*De+1.5*Pe):.2f}")

print("\nRESULT:", "ALL FORMAT CHECKS PASSED" if not fails else f"{len(fails)} FORMAT FAILURES: {fails}")
