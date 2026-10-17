"""Merge per-dataset filtered parts into one time-series table and one site table (native resolutions kept)."""
import glob
import os

import polars as pl

from common import PARTS, ROOT

ts = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(os.path.join(PARTS, '*.parquet')))
                if not f.endswith('_sites.parquet')])
sites = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(os.path.join(PARTS, '*_sites.parquet')))],
                  how='diagonal_relaxed')
sites = sites.join(ts.group_by('site_id').agg(pl.len().alias('n_rows'),
                                              pl.col('load_kw').is_not_null().mean().alias('load_coverage')),
                   on='site_id').sort('kwp', descending=True)
ts.write_parquet(os.path.join(ROOT, 'bigpv_timeseries.parquet'))
sites.write_parquet(os.path.join(ROOT, 'bigpv_sites.parquet'))
sites.write_csv(os.path.join(ROOT, 'bigpv_sites.csv'))
with pl.Config(tbl_cols=20, tbl_width_chars=250, fmt_str_lengths=40):
    print(sites.drop('note'))
print(f'{sites.height} sites, {ts.height:,} rows -> {ROOT}')
