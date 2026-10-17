"""Step 1: parse both consumption CSVs into cached daily-profile arrays (run once)."""
import sys, time
from config import LABELLED_TS, UNLABELLED_TS, CACHE_DIR
from data import load_timeseries, save_cache

jobs = {"labelled": LABELLED_TS, "unlabelled": UNLABELLED_TS}
which = sys.argv[1:] or list(jobs)
for name in which:
    t0 = time.time()
    print(f"== {name}: {jobs[name]}", flush=True)
    d = load_timeseries(jobs[name])
    save_cache(d, CACHE_DIR / f"{name}.npz")
    print(f"   done in {time.time() - t0:.0f}s")
    for k, v in d["stats"].items():
        print(f"   {k}: {v}")
