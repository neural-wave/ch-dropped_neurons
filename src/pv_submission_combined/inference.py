"""PV detection submission: one entry point, four tasks.

    python inference.py path/to/test_set.csv

  1. PV yes/no        cnn_daymodel: 2-D CNN on the customer's (days x hours) image
  2. install date     cnn_daymodel: day encoder + gradient-boosting day score, change point
  3. initial kWp      kWp network: per-day CNN + transformer around the install date
  4. extensions       pv_extension: weekly CNN + BiGRU (nothing / installed / extended)

Writes predictions.csv and pv_extensions_predictions.csv to the current working directory (UTF-8, ';', TRUE/FALSE,
YYYY-MM-DD, one row per customer of the test file). Only the consumption values and their timestamps are used.
See README.md.
"""
import importlib.util
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
COMP = HERE / 'components'
sys.path.insert(0, str(COMP / 'cnn_daymodel'))
warnings.filterwarnings('ignore', category=RuntimeWarning)

from data import date_to_day, load_timeseries, per_customer  # noqa: E402  (cnn_daymodel)
from features import capacity_features                       # noqa: E402
from pipeline import PVPipeline                               # noqa: E402

MAIN_COLS = ['customer_id', 'pv_installed', 'pv_installation_date', 'pv_capacity_kwp']
EXT_COLS = ['customer_id', 'extension_date', 'pv_capacity_after_extension_kwp']
PEAK_TO_KWP = 1.3          # cnn_daymodel fallback: kWp ~ 1.3 x summer midday peak reduction (kW)
CHUNK = 100


def load_module(name, path):
    """Import a component's inference.py under its own name (all three components call it inference.py)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def log(msg):
    print(f'[inference] {msg}', flush=True)


def detect_and_date(test_csv):
    """Tasks 1-2 with cnn_daymodel -> (all ids in file order, {cid: (pv, date str, fallback kWp)})."""
    pipe = PVPipeline.load(COMP / 'cnn_daymodel' / 'model')
    raw = load_timeseries(test_csv, verbose=False, require_pod=False)
    data = per_customer(raw)
    # output order = order of first appearance in the test file (ids kept as strings)
    sep = ';' if open(test_csv, encoding='utf-8').readline().count(';') else ','
    col = pd.read_csv(test_csv, sep=sep, usecols=['customer_id'], dtype=str)['customer_id'].dropna()
    ids = list(dict.fromkeys(c for c in col if c.strip()))
    out = {}
    cids = sorted(data)
    for i in range(0, len(cids), CHUNK):
        prep = pipe.prepare({c: data[c] for c in cids[i:i + CHUNK]})
        for _, r in pipe.predict(prep).iterrows():
            c, pv = r['customer_id'], bool(r['pv_installed'])
            date, fallback = (str(r['pv_installation_date'])[:10] if pv else ''), np.nan
            if pv:
                f = capacity_features(prep[c]['X'], prep[c]['days'], date_to_day(date), prep[c]['scale'], prep[c]['sun'])
                fallback = max(PEAK_TO_KWP * f['peak_red_kw'], 1.0)
            out[c] = (pv, date, fallback)
    return ids, out


def main():
    if len(sys.argv) != 2 or not Path(sys.argv[1]).is_file():
        raise SystemExit('usage: python inference.py path/to/test_set.csv')
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ids, det = detect_and_date(Path(sys.argv[1]))
    pv = {c: v for c, v in det.items() if v[0]}
    log(f'detection + date (cnn_daymodel): {len(ids)} customers, {len(pv)} PV  [{time.time() - t0:.0f}s]')

    kwp_mod = load_module('kwp_component', COMP / 'kwp' / 'inference.py')
    cons = kwp_mod.load_consumption(sys.argv[1])          # UTC day grid shared by the kWp and extension models
    kwp = {c: v[2] for c, v in pv.items()}                 # fallback first
    try:
        net, scale, post = kwp_mod.load_model(COMP / 'kwp' / 'kwp.pt', device)
        todo = [c for c in pv if c in cons]
        if todo:
            sites = [(*cons[c], kwp_mod.date_to_epoch_day(pv[c][1])) for c in todo]
            for c, k in zip(todo, np.exp(kwp_mod.predict_log_kwp(net, sites, post, device)) * scale):
                if np.isfinite(k) and k > 0:
                    kwp[c] = float(k)
        log(f'kWp (kWp network): {len(todo)} customers  [{time.time() - t0:.0f}s]')
    except Exception as e:
        log(f'WARNING kWp network failed ({type(e).__name__}); cnn_daymodel fallback kWp used')

    ext = pd.DataFrame(columns=EXT_COLS)
    try:
        ext_mod = load_module('extension_component', COMP / 'pv_extension' / 'inference.py')
        model, threshold = ext_mod.load_model(COMP / 'pv_extension' / 'ext_net.pt', device)
        ext = ext_mod.predict_extensions(model, threshold, {c: cons[c] for c in pv if c in cons}, kwp,
                                         {c: ext_mod.eday(pv[c][1]) for c in pv}, device=device)
        log(f'extensions (pv_extension): {len(ext)} event(s)  [{time.time() - t0:.0f}s]')
    except Exception as e:
        log(f'WARNING extension model failed ({type(e).__name__}); header-only extension file')

    rows = []
    for c in ids:
        p, date, _ = det.get(c, (False, '', np.nan))
        rows.append([c, 'TRUE' if p else 'FALSE', date if p else '', f'{kwp[c]:.2f}' if p else ''])
    pd.DataFrame(rows, columns=MAIN_COLS).to_csv('predictions.csv', sep=';', index=False, encoding='utf-8')
    ext[EXT_COLS].to_csv('pv_extensions_predictions.csv', sep=';', index=False, encoding='utf-8')
    log(f'wrote predictions.csv ({len(rows)} rows, {len(pv)} PV) and pv_extensions_predictions.csv '
        f'({len(ext)} rows) in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
