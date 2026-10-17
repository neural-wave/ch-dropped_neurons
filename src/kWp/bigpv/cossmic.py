"""CoSSMic / OPSD household_data (Konstanz, DE): 15-min cumulative kWh counters."""
import os

import polars as pl

from common import RAW, est_kwp, save, site_row

df = pl.read_csv(os.path.join(RAW, 'opsd/hh15.csv'), try_parse_dates=True, infer_schema_length=20000)
sites_all = sorted({c.split('_')[2] for c in df.columns if c.startswith('DE_KN_')})
parts, sites = [], []
for s in sites_all:
    cols = {c.split(f'{s}_')[1]: c for c in df.columns if c.startswith(f'DE_KN_{s}_')}
    pv_cols = [c for k, c in cols.items() if k.startswith('pv')]
    if not pv_cols or 'grid_import' not in cols:
        continue
    kw = lambda c: (pl.col(c).cast(pl.Float64).diff() * 4)  # counter at interval end -> mean kW over it
    x = df.select(pl.col('utc_timestamp').alias('t_end'),
                  pl.sum_horizontal([kw(c) for c in pv_cols]).alias('pv_kw'),
                  pl.all_horizontal([pl.col(c).is_not_null() for c in pv_cols]).alias('pv_ok'),
                  kw(cols['grid_import']).alias('grid_import_kw'),
                  (kw(cols['grid_export']) if 'grid_export' in cols else pl.lit(0.0)).alias('grid_export_kw'),
                  (kw(cols['storage_charge']) - kw(cols['storage_decharge'])
                   if 'storage_charge' in cols else pl.lit(0.0)).alias('storage_net_kw'))
    x = x.filter(pl.col('pv_ok') & pl.col('grid_import_kw').is_not_null())
    kwp = est_kwp(x['pv_kw'].to_numpy())
    # Without an export meter, import + pv is only gross load if the site never exports:
    # require it to keep importing (>= 20% of kWp) even in the sunniest intervals.
    sunny_import = x.filter(pl.col('pv_kw') > 0.5 * kwp)['grid_import_kw'].quantile(0.05)
    self_consumes = 'grid_export' in cols or sunny_import >= 0.2 * kwp
    x = x.with_columns(pl.when(self_consumes).then(pl.col('grid_import_kw') - pl.col('grid_export_kw') + pl.col('pv_kw')
                       - pl.col('storage_net_kw')).alias('load_kw'),
                       (pl.col('t_end') - pl.duration(minutes=15)).alias('timestamp'),
                       pl.lit('cossmic').alias('dataset'), pl.lit(s).alias('site_id'))
    parts.append(x)
    sector = 'industrial' if s.startswith('industrial') else s.rstrip('0123456789')
    if 'grid_export' in cols:
        note = 'load = import - export + pv - storage_net'
    elif self_consumes:
        note = f'no export meter; sunny-hour import p5={sunny_import:.1f} kW so export assumed 0; load = import + pv - storage_net'
    else:
        note = f'no export meter and sunny-hour import p5={sunny_import:.1f} kW: site exports, load_kw left null'
    sites.append(site_row('cossmic', s, kwp, 'estimated_p995', 15, x['timestamp'],
                          sector, 'DE', 'gross' if self_consumes else 'grid_import_only', note))
save('cossmic', pl.concat(parts, how='diagonal_relaxed'), pl.DataFrame(sites))
