"""BDG2: hourly kWh per building; solar meters only at site Bobcat (US). Meter may be net of PV."""
import os

import polars as pl

from common import RAW, est_kwp, save, site_row

D = os.path.join(RAW, 'bdg2/data/')
meta = pl.read_csv(D + 'metadata/metadata.csv', infer_schema_length=5000)
sol = pl.read_csv(D + 'meters/cleaned/solar_cleaned.csv', try_parse_dates=True)
ids = [c for c in sol.columns if c != 'timestamp' and sol[c].is_not_null().any()]
ele = pl.read_csv(D + 'meters/cleaned/electricity_cleaned.csv', try_parse_dates=True, columns=['timestamp'] + ids)
parts, sites = [], []
for b in ids:
    tz = meta.filter(pl.col('building_id') == b)['timezone'][0]
    m = meta.filter(pl.col('building_id') == b).row(0, named=True)
    df = (sol.select('timestamp', pl.col(b).cast(pl.Float64).alias('pv_kw'))
          .join(ele.select('timestamp', pl.col(b).cast(pl.Float64).alias('load_kw')), on='timestamp')
          .with_columns(pl.col('timestamp').dt.replace_time_zone(tz, ambiguous='earliest', non_existent='null'))
          .drop_nulls('timestamp').with_columns(pl.lit('bdg2').alias('dataset'), pl.lit(b).alias('site_id')))
    parts.append(df)
    sites.append(site_row('bdg2', b, est_kwp(df['pv_kw'].to_numpy()), 'estimated_p995', 60, df['timestamp'],
                          m['primaryspaceusage'], 'US', 'meter_unknown',
                          'electricity meter may be net of PV; hourly kWh treated as mean kW'))
save('bdg2', pl.concat(parts), pl.DataFrame(sites))
