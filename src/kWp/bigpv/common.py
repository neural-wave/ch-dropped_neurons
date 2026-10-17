"""Shared schema for the >10 kWp consumption+PV merge.

Per dataset we write /workspace/bigpv/parts/<dataset>.parquet (time series) and <dataset>_sites.parquet.
Time series columns (average kW over the interval starting at `timestamp`, UTC):
  dataset, site_id, timestamp, load_kw (gross on-site consumption), grid_import_kw, grid_export_kw, pv_kw
Site columns: dataset, site_id, kwp, kwp_source, resolution_min, start, end, sector, country, load_kind, note
  kwp_source: 'reported' (from dataset metadata) | 'estimated_p995' (99.5th pct of interval-mean PV kW)
  load_kind: 'gross' (measured/derived total consumption) | 'meter_unknown' (meter may be net of PV)
             | 'grid_import_only' (load_kw null; only grid import + pv known)
"""
import os

import numpy as np
import polars as pl

ROOT = '/workspace/bigpv'
PARTS = os.path.join(ROOT, 'parts')
RAW = os.path.join(ROOT, 'raw')
MIN_KWP = 10.0
TS_COLS = ['dataset', 'site_id', 'timestamp', 'load_kw', 'grid_import_kw', 'grid_export_kw', 'pv_kw']


def est_kwp(pv_kw):
    return float(np.nanpercentile(np.asarray(pv_kw, dtype=float), 99.5))


def save(name, ts, sites):
    os.makedirs(PARTS, exist_ok=True)
    ts = ts.with_columns(pl.col('timestamp').dt.convert_time_zone('UTC'),
                         *[pl.col(c).cast(pl.Float32) for c in TS_COLS[3:] if c in ts.columns])
    for c in TS_COLS[3:]:
        if c not in ts.columns:
            ts = ts.with_columns(pl.lit(None, pl.Float32).alias(c))
    ts = ts.select(TS_COLS).sort('site_id', 'timestamp')
    sites = sites.filter(pl.col('kwp') > MIN_KWP).with_columns(pl.col('start', 'end').dt.convert_time_zone('UTC'))
    ts = ts.filter(pl.col('site_id').is_in(sites['site_id'].to_list()))
    ts.write_parquet(os.path.join(PARTS, f'{name}.parquet'))
    sites.write_parquet(os.path.join(PARTS, f'{name}_sites.parquet'))
    print(sites)
    print(f'{name}: kept {sites.height} sites, {ts.height:,} rows')


def site_row(dataset, site_id, kwp, kwp_source, res, ts, sector, country, load_kind, note=''):
    return dict(dataset=dataset, site_id=site_id, kwp=kwp, kwp_source=kwp_source, resolution_min=res,
                start=ts.min(), end=ts.max(), sector=sector, country=country, load_kind=load_kind, note=note)
