"""Usage: python test/make_sample.py <dev_time_series.csv> <out.csv> <seed> [n]

Cuts n (default 5) randomly chosen customers out of the challenge time-series CSV, keeping the header, the `;` separator
and file order, so the result is a valid test_set.csv for inference.py.
"""
import csv, random, sys
src, dst, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
ids = []
with open(src, encoding="utf-8-sig", newline="") as f:
    r = csv.reader(f, delimiter=";"); header = next(r)
    seen = set()
    for row in r:
        if row[0] not in seen:
            seen.add(row[0]); ids.append(row[0])
n_pick = int(sys.argv[4]) if len(sys.argv) > 4 else 5
random.seed(seed)
pick = set(random.sample(ids, n_pick))
n = 0
with open(src, encoding="utf-8-sig", newline="") as f, open(dst, "w", encoding="utf-8", newline="") as g:
    r = csv.reader(f, delimiter=";"); w = csv.writer(g, delimiter=";", lineterminator="\n")
    w.writerow(next(r))
    for row in r:
        if row[0] in pick:
            w.writerow(row); n += 1
print("picked (file order):", [i for i in ids if i in pick], "rows:", n)
