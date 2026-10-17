"""Build a per-kWp 15-minute PV generation library from openclimatefix/uk_pv (5-minutely).

Used only to synthesise training targets; inference sees consumption alone.
Output (npz): gen[n_sys, n_days, 96] float16 kWh per 15 min per kWp (UTC slots, NaN = missing),
days (datetime64[D], UTC), ss_id, kwp.
"""
import argparse
import glob
import os

import numpy as np
import polars as pl

SNAP = glob.glob('/workspace/.hf_home/hub/datasets--openclimatefix--uk_pv/snapshots/*/')[0]


def load_month(year, month, meta, bad):
    path = os.path.join(SNAP, f'5_minutely/year={year}/month={month:02d}/data.parquet')
    if not os.path.exists(path):
        return None
    df = pl.read_parquet(path).join(meta.select('ss_id', 'kWp'), on='ss_id', how='inner')
    # Rows are period-ending; shift back so a 12:00 reading falls into the 11:45-12:00 slot.
    df = df.with_columns((pl.col('datetime_GMT') - pl.duration(minutes=5)).dt.truncate('15m').alias('t'))
    flagged = df.join_where(bad, pl.col('ss_id') == pl.col('ss_id_bad'),
                            pl.col('datetime_GMT') >= pl.col('bad_start'),
                            pl.col('datetime_GMT') <= pl.col('bad_end')).select('ss_id', 'datetime_GMT')
    df = df.join(flagged, on=['ss_id', 'datetime_GMT'], how='anti')
    agg = df.group_by('ss_id', 't').agg(pl.col('generation_Wh').sum().alias('wh'), pl.len().alias('n'),
                                        pl.col('kWp').first())
    # Instantaneous readings: require all three 5-min samples in the slot.
    return agg.filter(pl.col('n') == 3).with_columns((pl.col('wh') / 1000 / pl.col('kWp')).alias('kwh_per_kwp'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default='2022,2023,2024')
    ap.add_argument('--out', default='/workspace/data/pv_library_15min.npz')
    ap.add_argument('--min_coverage', type=float, default=0.6)
    args = ap.parse_args()

    meta = pl.read_csv(os.path.join(SNAP, 'metadata.csv')).filter(pl.col('kWp').is_between(0.5, 30))
    bad = pl.read_csv(os.path.join(SNAP, 'bad_data.csv'), try_parse_dates=True).select(
        pl.col('ss_id').alias('ss_id_bad'),
        pl.col('start_datetime_GMT').dt.cast_time_unit('ns').alias('bad_start'),
        pl.col('end_datetime_GMT').dt.cast_time_unit('ns').fill_null(
            pl.datetime(2100, 1, 1, time_unit='ns', time_zone='UTC')).alias('bad_end'))

    years = [int(y) for y in args.years.split(',')]
    days = np.arange(np.datetime64(f'{years[0]}-01-01'), np.datetime64(f'{years[-1] + 1}-01-01'))
    parts = []
    for y in years:
        for m in range(1, 13):
            part = load_month(y, m, meta, bad)
            if part is not None:
                parts.append(part.select('ss_id', 't', 'kwh_per_kwp'))
                print(y, m, part.height, flush=True)
    df = pl.concat(parts)
    ids = np.sort(df['ss_id'].unique().to_numpy())
    sys_idx = {s: i for i, s in enumerate(ids)}
    gen = np.full((len(ids), len(days), 96), np.nan, dtype=np.float16)
    t = df['t'].dt.replace_time_zone(None).to_numpy()
    d = (t.astype('datetime64[D]') - days[0]).astype(int)
    slot = ((t - t.astype('datetime64[D]')).astype('timedelta64[m]').astype(int)) // 15
    si = np.vectorize(sys_idx.get)(df['ss_id'].to_numpy())
    # Clip physically implausible spikes (>1.3 kW per kWp sustained for 15 min).
    gen[si, d, slot] = np.clip(df['kwh_per_kwp'].to_numpy(), 0, 0.325)

    # Keep system-days with full daylight coverage, and systems with enough such days.
    day_ok = np.isfinite(gen).mean(axis=2) > 0.95
    gen[~day_ok] = np.nan
    yield_day = np.nansum(gen.astype(np.float32), axis=2)
    keep = day_ok.mean(axis=1) >= args.min_coverage
    # Drop systems whose mean daily yield per kWp is implausible (mis-specified kWp).
    mean_yield = np.where(day_ok, yield_day, np.nan)
    mean_yield = np.nanmean(mean_yield, axis=1)
    keep &= (mean_yield > 1.0) & (mean_yield < 4.5)
    print(f'systems kept {keep.sum()}/{len(ids)}; median daily yield/kWp {np.nanmedian(mean_yield[keep]):.2f}')
    kwp = meta.filter(pl.col('ss_id').is_in(ids[keep].tolist())).sort('ss_id')['kWp'].to_numpy()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, gen=gen[keep], days=days, ss_id=ids[keep], kwp=kwp)
    print('saved', args.out, gen[keep].shape)


if __name__ == '__main__':
    main()
